import argparse
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, automatic_layer_slicing)
from parax.model.moe import FlaxMoEForLMModule, MoEConfig, TrainState
from parax.model.model_util import optax_adafactor
from parax.util import (run_cmd, write_tsv, map_to_shape, list_gpu_info, benchmark_func,
                        count_communication_primitives, print_used_time,
                        compute_param_number)

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
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        expert_group_size, expert_number,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        prefer_reduce_scatter, use_remat = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    if num_micro_batches > 1:
        grad_func = parax.grad
        prefer_reduce_scatter = False
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
        expert_number=expert_number
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

    # Restore global config
    global_config.restore(backup)

    return latencies, alloc_mem, tflops, param_count, ilp_objective


def benchmark_one_case(case):
    # Launch physical mesh
    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()

    # Run benchmark
    result = benchmark_moe_internal(physical_mesh, case, args.niter)
    latencies, alloc_mem, tflops, param_count, ilp_objective = result

    # Log results
    heads = ["Model", "Model Config", "Parallel Config", "Param Count",
             "Alloc Mem", "ILP Objective", "Mean Latency", "Std Latency", "TFLOPS"]
    values = ["moe", case[:-6], case[-6:],
              f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{ilp_objective:.2f}",
              f"{np.mean(latencies):.3f}", f"{np.std(latencies):.3f}", f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_moe.tsv")

    physical_mesh.shutdown()


# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, S_ = expert_group_size, E = expert_number,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel,
# RS = prefer_reduce_scatter, CK = use_checkpoint,

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (8,   1024, 1024, 12, 1024//64,  25600, 1024, 4,  1,  1,  1,  False, True,  False),
],

2: [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (16,  1024, 1280, 12, 1280//128, 25600, 1024, 16, 1,  2,  1,  True,  True,  False),
],

4: [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (16,  1024, 1024, 6,  1024//128, 25600, 1024, 8,  1,  4,  1,  False, False, False),
],


8: [
    #B,   S,    H,    L,  #head,     V,     S_,   E,  D0, D1, NB, FD,    RS,    CK
    (32,  1024, 1024, 6,  1024//128, 25600, 1024, 8,  2,  4,  1,  False, True,  False),
    (32,  1024, 1024, 6,  1024//128, 25600, 1024, 8,  2,  4,  1,  False, False, False),
],

16: [
    #(32,  1024, 2560, 12, 2560//64, 25600, 1024, 16, 2,  8,  1,  False, False, False),
    (128,  1024, 2560, 12, 2560//64, 25600, 1024, 16, 2,  8,  4,  False, False, False),
    (256,  1024, 2560, 12, 2560//64, 25600, 1024, 16, 2,  8,  8,  False, False, False),
]

}


benchmark_suites = {
    "default": default_benchmark_suite,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    # Set global environments
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')
        num_gpus = int(ray.cluster_resources()["GPU"])

    global_config.use_dummy_value_for_benchmarking = True

    # Get benchmark suite and run all cases
    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        exit()

    for case in suite:
        benchmark_one_case(case)
