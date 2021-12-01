import argparse
import pickle
from datetime import datetime

import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import ray

import parax
from benchmark.util import compute_moe_parameter_count
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, automatic_layer_slicing)
from parax.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from parax.model.model_util import optax_adafactor
from parax.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info, benchmark_func,
                        count_communication_primitives, print_used_time,
                        compute_param_number)

from benchmark.parax.paper_manual_moe_suite import paper_moe_suite, test_moe_suite
from benchmark_gpt_bert import load_profiling_result, get_train_step


GB = 1024 ** 3


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


def benchmark_moe_internal(physical_mesh, benchmark_case, niter):
    # Backup global config
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size, \
        num_experts, expert_group_size, \
        mesh_dim0, mesh_dim1, _, _ , _, num_micro_batches, force_data_parallel,\
        use_remat, prefer_reduce_scatter, _, _ = benchmark_case
    dtype = jnp.float16

    expected_expert_group_size = batch_size * seq_len // num_micro_batches \
                        // mesh_dim0 // 2
    if expected_expert_group_size != expert_group_size:
        print("- Expected expert group size should be {}, but got {}".
              format(expected_expert_group_size, expert_group_size))
        expert_group_size = expected_expert_group_size

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = parax.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    global_config.force_data_parallel = force_data_parallel
    global_config.prefer_reduce_scatter = prefer_reduce_scatter
    global_config.allow_mixed_mesh_shape = True

    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

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
        intermediate_size=hidden_size * 8,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        expert_group_size=expert_group_size,
        expert_number=num_experts,
        gradient_checkpointing=use_remat
    ), dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
    param_count = compute_param_number(state.params)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, dtype)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Check sharding strategy
    alloc_mem = executable.get_total_allocation_size()
    ilp_objective = testing.last_compiled_auto_sharding_objective or 0.0
    hlo_text = executable.get_hlo_text()
    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")

    # Benchmark step time
    if alloc_mem > 26 * GB: # out of memory
        latencies = [-1]
    else:
        for i in range(niter):
            state = train_step(state, batch, rngkey)

        latencies = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    # Compute statistics
    num_gpus = mesh_dim0 * mesh_dim1
    tflops = executable.flop_count / num_gpus / np.mean(latencies) / 1e12
    parameter_count = compute_moe_parameter_count(num_layers, hidden_size, vocab_size, num_experts,
                                                  mlp_factor=8)
    peak_mem = physical_mesh.get_max_memory_allocated()

    # Restore global config
    global_config.restore(backup)

    return parameter_count, ilp_objective, peak_mem, latencies, tflops


TMP_PICKLE_FILE_NAME = "tmp/tmp_transfer_moe.pkl"


def benchmark_one_case(case, niter, local, use_separate_process=False, dump_result=False):
    if not use_separate_process:
        # Launch physical mesh
        if local:
            physical_mesh = PhysicalDeviceMesh(jax.devices())
        else:
            ray.init(address="auto", ignore_reinit_error=True)
            device_cluster = DeviceCluster()
            physical_mesh = device_cluster.get_physical_mesh()
            jax.config.update('jax_platform_name', 'cpu')

        global_config.use_dummy_value_for_benchmarking = True

        # Run benchmark
        result = benchmark_moe_internal(physical_mesh, case, niter)
        physical_mesh.shutdown()
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        ret = run_cmd("python3 benchmark_moe_2d_one_case.py "
                      f"--niter {niter} "
                      f'--case "{case}" '
                      f"{'--local' if local else ''} "
                      f"--dump-result ")
        if ret == 0:
            result = pickle.load(open(TMP_PICKLE_FILE_NAME, "rb"))
        else:
            result = -1, -1, -1, [-1], -1

    if dump_result:
        pickle.dump(result, open(TMP_PICKLE_FILE_NAME, "wb"))

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--case", type=str, required=True)
    parser.add_argument("--local", action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--dump-result", action="store_true",
                        help="Dump results into a temporary pickle file")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(case, args.niter, args.local, False, args.dump_result)
