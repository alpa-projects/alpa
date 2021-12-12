import argparse
from functools import partial
import os
import pickle
import time

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ray

import parax
from benchmark.util import compute_gpt_parameter_count, compute_gpt_tflops, GB
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule, TrainState
from parax.model.gpt_model import FlaxGPTForLMModule
from parax.util import map_to_shape, count_communication_primitives, print_used_time, run_cmd


def load_profiling_result(physical_mesh):
    filename = physical_mesh.get_signature() + ".prof.pkl"
    if os.path.exists(filename):
        print(f"Load saved profiling results from {filename}")
        physical_mesh.load_profiling_result(filename)
        physical_mesh.prof_result.make_monotonic()
        physical_mesh.prof_result.multiply_scale(1e7)
    else:
        physical_mesh.profile_collective("all-reduce")
        print(f"Save profiling results to {filename}")
        physical_mesh.save_profiling_result(filename)


def create_train_state(rngkey, model, dtype, batch):
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])

    def weight_decay_mask(pytree):
        # do not use weight decay on layer norm and bias.
        return jax.tree_map(lambda x: x.ndim > 1, pytree)

    tx = optax.chain(
        #optax.clip_by_global_norm(1.0),  # TODO(lmzheng): fix reduce-scatter for this
        optax.adamw(learning_rate=1e-2, mask=weight_decay_mask)
    )

    mixed_precision = (dtype == jnp.float16)

    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        mixed_precision=mixed_precision,
        dynamic_scale=None)
    return state
 

def get_train_step(grad_func, num_layers, dtype):

    @parallelize
    def train_step(state, batch, rng_key):
        def loss_func(params):
            rngs = {"dropout": rng_key}
            logits = state.apply_fn(params,
                                    batch["input_ids"],
                                    batch["attention_mask"],
                                    batch["token_type_ids"],
                                    batch["position_ids"],
                                    deterministic=True,
                                    rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grads = grad_func(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        # TODO(lmzheng): add dynamic scaling for mixed-precision training
        return new_state

    return train_step


def benchmark_gpt_bert_internal(physical_mesh, model_type, benchmark_case, niter):
    # Backup global config
    backup = global_config.backup()
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        mesh_dim0, mesh_dim1, _, _, _,  num_micro_batches, force_batch_dim_mapping,\
        use_remat, prefer_reduce_scatter, other, overwrite_global_config_dict = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = parax.grad
    else:
        num_micro_batches = None
        grad_func = jax.grad

    if force_batch_dim_mapping:
        # Always map batch dim to mesh dim 0
        global_config.force_batch_dim_to_mesh_dim = 0
    global_config.prefer_reduce_scatter = prefer_reduce_scatter

    if other == "zero-3":
        global_config.force_zero_stage_3 = True
    elif other in ["shard-largest"]:
        global_config.force_simple_heuristic = other
        global_config.remat_using_while = True

    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1])
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

    print_used_time("Setup device mesh")

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
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads,
            vocab_size=vocab_size,
            max_position_embeddings=seq_len,
            type_vocab_size=0,
            gradient_checkpointing=use_remat,
        ), dtype=dtype)
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, dtype, batch)
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
    with open("tmp/last_2d.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)

    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")
    print(f"alloc_mem: {alloc_mem / GB:.2f} GB")

    # Benchmark step time
    warmup = 2 if niter >= 5 else 1

    if alloc_mem > physical_mesh.get_available_memory():
        latencies = [-1]
    else:
        for i in range(niter):
            state = train_step(state, batch, rngkey)

        latencies = executable.get_execution_time_costs(warmup=warmup)
    print_used_time("Benchmark")

    # Compute statistics
    tflops = compute_gpt_tflops(batch_size, seq_len, num_layers,
                                hidden_size, vocab_size,
                                physical_mesh.total_devices,
                                np.mean(latencies), use_remat)
    param_count = compute_gpt_parameter_count(num_layers, hidden_size, vocab_size)
    peak_mem = physical_mesh.get_max_memory_allocated()

    # Restore global config
    global_config.restore(backup)

    return param_count, ilp_objective, peak_mem, latencies, tflops


TMP_PICKLE_FILE_NAME = "/tmp/tmp_transfer.pkl"


def benchmark_one_case(model, case, niter,
                       num_hosts, num_devices_per_host,
                       local, use_separate_process,
                       dump_result=False):
    if not use_separate_process:
        # Launch physical mesh
        if local:
            assert num_hosts == 1
            physical_mesh = PhysicalDeviceMesh(jax.devices()[:num_devices_per_host])
        else:
            ray.init(address="auto", ignore_reinit_error=True)
            device_cluster = DeviceCluster()
            physical_mesh = device_cluster.get_physical_mesh(
                list(range(num_hosts)), num_devices_per_host)
            jax.config.update('jax_platform_name', 'cpu')

        global_config.use_dummy_value_for_benchmarking = True

        # Run benchmark
        result = benchmark_gpt_bert_internal(physical_mesh, model, case, niter)

        physical_mesh.shutdown()
    else:
        # Launch a new process for benchmark to isolate errors.
        # Get the return data via pickle.
        run_cmd(f"rm -rf {TMP_PICKLE_FILE_NAME}")
        ret = run_cmd("python3 benchmark_gpt_bert_2d_one_case.py "
                     f"--model {model} "
                     f"--niter {niter} "
                     f'--case "{case}" '
                     f"--num-hosts {num_hosts} "
                     f"--num-devices-per-host {num_devices_per_host} "
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
    parser.add_argument("--model", type=str)
    parser.add_argument("--niter", type=int)
    parser.add_argument("--case", type=str)
    parser.add_argument("--num-hosts", type=int)
    parser.add_argument("--num-devices-per-host", type=int)
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    parser.add_argument("--dump-result", action="store_true",
        help="Dump results into a temporary pickle file")
    args = parser.parse_args()

    run_cmd("mkdir -p tmp")
    case = eval(args.case)
    benchmark_one_case(args.model, case, args.niter, args.num_hosts, args.num_devices_per_host,
                       args.local, False, args.dump_result)
