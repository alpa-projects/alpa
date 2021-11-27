import pickle

from datetime import datetime
import time

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import parax
from benchmark_gpt_bert_3d_one_case import get_train_step
from benchmark.util import compute_moe_parameter_count, compute_moe_tflops, run_cmd
from parax import (global_config, set_parallelize_options, DeviceCluster)
from parax.model.model_util import optax_adafactor
from parax.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from parax.util import (write_tsv, print_used_time)
from benchmark.parax.paper_manual_moe_suite import test_moe_suite, paper_moe_suite

GB = 1024 ** 3


benchmark_suites = {
    "test_moe": test_moe_suite,
    "paper_moe": paper_moe_suite
}



def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax_adafactor(
        learning_rate=1e-2, weight_decay_mask=weight_decay_mask
    )

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mixed_precision = (dtype == jnp.float16),
        dynamic_scale=None)
    return state


def benchmark_moe_internal(benchmark_case, niter):
    # Backup global config
    global time
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, num_experts, expert_group_size, \
        l_dim0, l_dim1, p_dim0, p_dim1, pipeline_mp_size,\
        num_micro_batches, force_data_parallel, use_remat, tie_word_embeddings, \
        auto_layer, _ = benchmark_case
    dtype = jnp.float16

    prefer_reduce_scatter = False

    # Parallel configs
    grad_func = parax.grad
    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter

    device_cluster = DeviceCluster()
    virtual_mesh = device_cluster.get_virtual_mesh()
    set_parallelize_options(devices=virtual_mesh,
                            strategy="3d_parallel",
                            num_micro_batches=num_micro_batches,
                            sub_physical_mesh_shapes=[(p_dim0, p_dim1)] * pipeline_mp_size,
                            sub_logical_mesh_shapes=[(l_dim0, l_dim1)] * pipeline_mp_size,
                            pipeline_parallel_schedule="1f1b")

    # Prepare input batch
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
    }
    print_used_time("Prepare input")

    # Init train state
    model = FlaxMoEForLMModule(MoEConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 8, # this is specific to gspmd.
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        expert_group_size=expert_group_size,
        expert_number=num_experts,
        pipeline_mp_size=pipeline_mp_size,
        tie_word_embeddings=tie_word_embeddings,
        gradient_checkpointing=use_remat
    ), dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, use_remat, pipeline_mp_size, dtype)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    # dump hlo ir for debugging
    stage_hlo_texts = executable.get_hlo_text()
    for i in range(len(stage_hlo_texts)):
        with open(f"tmp/stage_{i}.hlo", "w") as fout:
            fout.write(stage_hlo_texts[i])
    with open(f"tmp/resharding_tasks.txt", "w") as fout:
        fout.write(executable.print_resharding_tasks())

    executable.sync()
    print_used_time("Compile (worker)")

    for i in range(niter):
        state = train_step(state, batch, rngkey)

    timer_name = "overall"
    latencies = executable.get_execution_time_costs(warmup=0, timer_name=timer_name)[2:]
    print_used_time("Benchmark")

    mem_allocated = executable.get_memory_allocated()
    max_mem_allocated = executable.get_max_memory_allocated()

    # Compute statistics
    tflops = compute_moe_tflops(batch_size, seq_len, num_layers,
                                hidden_size, expert_group_size, vocab_size, num_experts,
                                virtual_mesh.total_devices, np.mean(latencies))
    tflops_ckpt = compute_moe_tflops(batch_size, seq_len, num_layers,
                                     hidden_size, expert_group_size, vocab_size, num_experts,
                                     virtual_mesh.total_devices, np.mean(latencies),
                                     checkpoint_activations=True)
    parameter_count = compute_moe_parameter_count(num_layers, hidden_size, vocab_size, num_experts,
                                                  mlp_factor=8)
    # Restore global config
    global_config.restore(backup)
    executable.shutdown()
    return parameter_count, mem_allocated, max_mem_allocated, latencies, tflops, tflops_ckpt


TMP_PICKLE_FILE_NAME = "tmp/tmp_transfer.pkl"


def benchmark_one_case(case, niter, use_separate_process=False, dump_result=False):
    if not use_separate_process:
        ray.init(address="auto", ignore_reinit_error=True)
        jax.config.update('jax_platform_name', 'cpu')
        global_config.use_dummy_value_for_benchmarking = True

        result = benchmark_moe_internal(case, niter)
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        ret = run_cmd("python3 benchmark_moe_3d_one_case.py "
                      f"--niter {niter} "
                      f'--case "{case}" '
                      f"--dump-result ")
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, -1, [-1], -1, -1

    if dump_result:
        pickle.dump(result, open(TMP_PICKLE_FILE_NAME, "wb"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=7)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--dump-result", action="store_true",
                        help="Dump results into a temporary pickle file")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(case, args.niter,
                       use_separate_process=False, dump_result=args.dump_result)
