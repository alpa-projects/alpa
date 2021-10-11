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


def create_train_state(rngkey, model, batch):
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
        dynamic_scale=None)
    return state


def benchmark_model_one_case(benchmark_case):
    print_used_time(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        expert_group_size, expert_number,\
        mesh_dim0, mesh_dim1, num_micro_batches, force_data_parallel,\
        use_remat = benchmark_case
    dtype = jnp.float16

    # Parallel configs
    global_config.force_data_parallel = force_data_parallel

    if num_micro_batches > 1:
        grad_func = parax.grad
        global_config.prefer_reduce_scatter = False
    else:
        num_micro_batches = None
        grad_func = jax.grad
        global_config.prefer_reduce_scatter = True

    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh, num_micro_batches=num_micro_batches)

    # Load profiling results
    if args.use_profiling:
        load_profiling_result(physical_mesh)
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
    model = FlaxMoEForLMModule(MoEConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 8,
        num_attention_heads=num_heads,
        max_position_embeddings=seq_len,
        vocab_size=vocab_size,
        expert_group_size=expert_group_size,
        expert_number=expert_number,
    ), dtype=dtype)

    rngkey = jax.random.PRNGKey(0)
    state = create_train_state(rngkey, model, batch)
    param_count = compute_param_number(state.params)
    print_used_time("Create train state")

    # Compile executable
    train_step = get_train_step(grad_func, num_layers, use_remat, dtype)
    executable = train_step.get_executable(state, batch, rngkey)
    print_used_time("Compile (driver)")

    physical_mesh.sync_workers()
    print_used_time("Compile (workers)")

    # Benchmark step time
    for i in range(args.niter):
        state = train_step(state, batch, rngkey)

    costs = executable.get_execution_time_costs(warmup=2)
    print_used_time("Benchmark")

    # Check sharding strategy
    objective = testing.last_compiled_auto_sharding_objective or 0.0
    alloc_mem = executable.get_total_allocation_size()
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, n_all_to_all =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}, "
          f"#all-to-all: {n_all_to_all}")

    # Log benchmark results
    num_gpus = mesh_dim0 * mesh_dim1
    tflops = executable.flop_count / num_gpus / np.mean(costs) / 1e12
    heads = ["Type", "Model Config", "Parallel Config", "Param Count",
             "Alloc Mem", "ILP Objective", "Mean Time", "Std Time", "TFLOPS"]
    values = ["moe", str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{param_count/1e9:.3f}", f"{alloc_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}", f"{tflops:.2f}"]
    write_tsv(heads, values, f"result_moe.tsv")

    physical_mesh.shutdown()



# B = batch_size, S = seq_len, H = hidden_size, L = num_layers, V = vocab_size
# #head = num_heads, S_ = expert_group_size, E = expert_number,
# D0 = mesh_dimension_0, D1 = mesh_dimension_1,
# NB = num_micro_batches, FD = force_data_parallel, CK = use_checkpoint

default_benchmark_suite = {  # key = number of gpus, value = a list of cases
1: [
    #B,   S,    H,    L,  #head,    V,     S_,   E,  D0, D1, NB, FD,    CK
    (8,   1024, 1024, 12, 1024//64, 25600, 1024, 4,  1,  1,  1,  False, False),
],

4: [
],

8: [
    (16,   1024, 1024, 12, 1024//64, 25600, 1024, 8,  8,  1,  1,  False, False),
],

16: [
]
}


benchmark_suites = {
    "default": default_benchmark_suite,
}

def benchmark_all():
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    try:
        suite = benchmark_suites[args.suite][num_gpus]
    except KeyError:
        suite = None

    if not suite:
        print(f"No available benchmark suite for {args.suite} on {num_gpus} GPUs")
        return

    for case in suite:
        # Backup global config
        backup = global_config.backup()

        benchmark_model_one_case(case)

        # Restore global config
        global_config.restore(backup)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=10,
        help="Number of benchmark iteration")
    parser.add_argument("--suite", choices=["default"], default="default")
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all()

