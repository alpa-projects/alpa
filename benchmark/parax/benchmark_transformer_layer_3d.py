import argparse
import gc
import os
import time

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

from parax import (parallelize, global_config, set_parallelize_options, testing,
                   DeviceCluster, PhysicalDeviceMesh, mark_pipeline)
from parax.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection
from parax.util import run_cmd, write_tsv, benchmark_func, list_gpu_info

import timeit

MB = 1024 ** 2
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
    batch_size, seq_len, hidden_size, num_layers, num_heads, dp_size, \
    tensor_mp_size, pipeline_mp_size, num_micro_batches, force_data_parallel \
    = benchmark_case
    is_pipeline_parallel = True if pipeline_mp_size > 1 else False
    dtype = jnp.float32
    if force_data_parallel:
        global_config.force_batch_dim_to_mesh_dim = 0
        global_config.allow_all_gather = False

    # Mesh configs
    if args.local:
        physical_mesh = PhysicalDeviceMesh(jax.devices())
    else:
        device_cluster = DeviceCluster()
        virtual_mesh = device_cluster.get_virtual_mesh()
        # virtual_mesh = virtual_mesh.slice(1, [[0, 1, 2, 3]])
    #     physical_mesh = device_cluster.get_physical_mesh()
    # logical_mesh = physical_mesh.get_logical_mesh([dp_size, tensor_mp_size],
    #                                               mesh_topology="tree",
    #                                               inter_host_bandwidth=1,
    #                                               intra_host_bandwidth=30)
    set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")

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
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params, hidden_states):
            rngs = {"dropout": batch["rng"]}
            if pipeline_mp_size:
                hidden_states, = mark_pipeline(hidden_states, name="0", mark_type="start")
            out = apply_fn(params, hidden_states, batch["attention_mask"],
                           deterministic=False, rngs=rngs)[0]
            ret = jnp.mean((out - batch["label"]) ** 2)
            if pipeline_mp_size:
                ret, = mark_pipeline(ret, name=str(pipeline_mp_size - 1), mark_type="end")
            return ret

        grad, grad_x = jax.grad(loss_func, argnums=(0, 1))(optimizer.target, batch["hidden_states"])
        # new_optimizer = optimizer.apply_gradient(grad)
        return grad

    # Prepare input batch
    batch = {
        "hidden_states": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=np.int32),
        "label": jnp.ones((batch_size, seq_len, hidden_size), dtype=np.float32),
        "rng": jax.random.PRNGKey(0),
    }
    log_time_stamp("Prepare input")

    # Init model and optimizer
    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads,
        pipeline_mp_size=pipeline_mp_size))
    rngkey = jax.random.PRNGKey(0)
    params = model.init_dummy(rngkey, batch["hidden_states"], batch["attention_mask"])
    optimizer = optim.Adam(1e-2).create(params)
    del (params, rngkey)
    log_time_stamp("Init model and optimizer")

    # Compile executable
    executable = train_step.get_executable(optimizer, batch, model.apply)
    log_time_stamp("Compile (driver)")

    # Benchmark step time
    def run_func():
        nonlocal optimizer
        # optimizer = train_step(optimizer, batch, model.apply)
        train_step(optimizer, batch, model.apply)

    def sync_func():
        if not is_pipeline_parallel:
            physical_mesh.sync_workers()
        return


    costs = benchmark_func(run_func, sync_func,
                           warmup=1, repeat=2, number=args.number)
    log_time_stamp("Benchmark")

    # Check sharding strategy
    if not is_pipeline_parallel:
        real_mem = physical_mesh.get_total_allocation_size(executable)
        objective = testing.last_compiled_auto_sharding_objective or 0.0
        hlo_module = testing.last_compiled_executable.hlo_modules()[0]
        hlo_ir = hlo_module.to_string()
        print(f" - #comm {hlo_ir.count('channel_id')}, " +
              f"#all-reduce {hlo_ir.count('all-reduce(') + hlo_ir.count('all-reduce-start(')}")

        with open("last.hlo", "w") as fout:
            fout.write(hlo_ir)
    else:
        real_mem = -1
        objective = -1

    # Log benchmark results
    heads = ["Type", "Case", "Mesh Shape", "Peak Mem", "Objective", "Mean Time", "Std Time"]
    values = ["transformer-layer", str(benchmark_case[:-5]), str(benchmark_case[-5:]),
              f"{real_mem/GB:.3f}", f"{objective:.2f}",
              f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, "result_trans.tsv")

    # physical_mesh.shutdown()

# B = batch_size, S = seq_len, H = hidden_size, L = num_layers,
# #head = num_heads, D1 = mesh_dimension_1, D2 = mesh_dimension_2

benchmark_suite_4_gpu = [
    # # B,  S,    H,    L,  #head,     D1, D2, PP, NB, FD
    (32,  1024, 1536, 2,  1536//96,  1,  2, 2, 1, False),
    (32,  1024, 1536, 2,  1536//96,  2,  1, 2, 1, False),
    (32,  128,  5120, 2,  5120//128, 1,  2, 2, 1, False),
    (32,  128,  5120, 2,  5120//128, 2,  1, 2, 1, False),
    (32,  1024, 1536, 2,  1536//96,  2,  1, 2, 1, False),
    (32,  1024, 1536, 2,  1536//96,  2,  1, 2, 1, False),
    (32,  1024, 1536, 4,  1536//96,  2,  1, 2, 1, False),
    (32,  1024, 1536, 4,  1536//96,  2,  1, 4, 1, False),
    (32,  1024, 1536, 2,  1536//96,  2,  1, 2, 1, False),
     # (24,  1024, 1536, 4,  1536//96,  2,  1, 4, 1, False),
]

benchmark_suite_8_gpu = [
    # B,  S,    H,    L,  #head,     D1, D2, PP, NB, FD
    # (32,  1024, 1536, 2,  1536//96,  4,  1, 2, 1, False),
    (16,  1024, 1536, 2,  1536//96,  4,  1, 2, 1, False),
    #
    # (32,  128,  5120, 2,  5120//128, 1,  4, 2, 1, False),
    # (32,  128,  5120, 2,  5120//128, 4,  1, 2, 1, False),
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
        benchmark_transformer_one_case(case, use_profiling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--number", type=int, default=5)
    parser.add_argument("--local", action="store_true",
                        help="Run on local GPUs. Do not use ray actors.")
    args = parser.parse_args()

    if not args.local:
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)
