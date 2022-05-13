import argparse
import copy
import os
import time

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

from alpa import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, LocalPhysicalDeviceMesh)
from alpa.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection
from alpa.util import (run_cmd, write_tsv, benchmark_func, list_gpu_info,
                        count_communication_primitives)

import timeit

GB = 1024 ** 3


tic = time.time()
def log_time_stamp(message):
    global tic
    if message:
        print(f" - {message}: {time.time() - tic:.2f} s")
    tic = time.time()


def compute_data_parallel_cost(optimizer, logical_mesh, physical_mesh):
    """For debugging usage."""
    shapes = jax.tree_util.tree_map(lambda x : np.prod(x.shape), optimizer.target)
    sizes = jax.tree_util.tree_leaves(shapes)
    cost = 0
    print(logical_mesh.mesh_beta)
    for size in sizes:
        cost += logical_mesh.all_reduce_cost(size * 4, 0)
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,4), (1,5), (2,6), (3,7),), size / 4, "float32")
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,2,4,6,), (1,3,5,7)), size / 2, "float32")
        #cost += physical_mesh.prof_result.estimate_all_reduce(((0,1,2,3,4,5,6,7),), size, "float32")
    print(cost)


def benchmark_transformer_one_case(benchmark_case, use_profiling):
    log_time_stamp(None)

    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, mesh_dim0, mesh_dim1 =\
        benchmark_case

    # Parallel configs
    if args.local:
        physical_mesh = LocalPhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([mesh_dim0, mesh_dim1],
                                                  mesh_topology="tree",
                                                  inter_host_bandwidth=1,
                                                  intra_host_bandwidth=30)
    set_parallelize_options(devices=logical_mesh)

    # Load profiling results
    if use_profiling:
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
    log_time_stamp("Setup device mesh")

    @parallelize
    def train_step(optimizer, batch, rng_key, apply_fn):
        def loss_func(params):
            rngs = {"dropout": rng_key}
            out = apply_fn(params, batch["hidden_states"], batch["attention_mask"],
                           deterministic=False, rngs=rngs)[0]
            return jnp.mean((out - batch["label"]) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Prepare input batch
    batch = {
        "hidden_states": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=np.int32),
        "label": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
    }
    log_time_stamp("Prepare input")

    # Init model and optimizer
    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads))
    rngkey = jax.random.PRNGKey(0)
    params = model.init_dummy(rngkey, batch["hidden_states"], batch["attention_mask"])
    optimizer = optim.Adam(1e-2).create(params)
    del params
    log_time_stamp("Init model and optimizer")

    # Compile executable
    executable = train_step.get_executable(optimizer, batch, rngkey, model.apply)
    log_time_stamp("Compile (driver)")

    physical_mesh.sync_workers()
    log_time_stamp("Compile (workers)")

    # Benchmark step time
    for i in range(args.niter):
        optimizer = train_step(optimizer, batch, rngkey, model.apply)

    costs = executable.get_execution_time_costs(warmup=2)
    log_time_stamp("Benchmark")

    # Check sharding strategy
    objective = testing.last_compiled_auto_sharding_objective or 0.0
    real_mem = executable.get_total_allocation_size()
    hlo_text = executable.get_hlo_text()

    with open("last.hlo", "w") as fout:
        fout.write(hlo_text)
    n_total, n_all_reduce, n_all_gather, n_reduce_scatter, _ =\
        count_communication_primitives(hlo_text)
    print(f"#total: {n_total}, #all-reduce: {n_all_reduce}, "
          f"#all-gather: {n_all_gather}, #reduce-scatter: {n_reduce_scatter}")

    # Log benchmark results
    heads = ["Type", "Model Config", "Parallel Config", "Peak Mem",
             "Objective", "Mean Time", "Std Time"]
    values = ["transformer-layer", str(benchmark_case[:-2]), str(benchmark_case[-2:]),
             f"{real_mem/GB:.3f}", f"{objective:.2f}",
             f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, "result_trans.tsv")

    physical_mesh.shutdown()

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, D0 = mesh_dimension_0, D1 = mesh_dimension_1

benchmark_suite_4_gpu = [
    # B,  S,    H,    L,  #head,     D0, D1
    (32,  1024, 1536, 2,  1536//96,  4,  1),
    (32,  1024, 1536, 2,  1536//96,  2,  2),
    (32,  1024, 1536, 2,  1536//96,  1,  4),
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     D0, D1
    (32,  1024, 1536, 4,  1536//96,  8,  1),
    (32,  1024, 1536, 4,  1536//96,  4,  2),
    (32,  1024, 1536, 4,  1536//96,  2,  4),

    (32,  128,  5120, 3,  5120//128, 8,  1),
    (32,  128,  5120, 3,  5120//128, 4,  2),
    (32,  128,  5120, 3,  5120//128, 2,  4),
]

def benchmark_all(use_profiling):
    if args.local:
        num_gpus = list_gpu_info().count("UUID")
    else:
        num_gpus = int(ray.cluster_resources()["GPU"])

    benchmark_suites = {
        4: benchmark_suite_4_gpu,
        8: benchmark_suite_8_gpu,
    }

    for case in benchmark_suites[num_gpus]:
        # Backup global config
        old_global_config = copy.deepcopy(global_config.__dict__)

        benchmark_transformer_one_case(case, use_profiling)

        # Restore global config
        global_config.__dict__ = old_global_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--niter", type=int, default=10)
    parser.add_argument("--local", action="store_true",
        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)
