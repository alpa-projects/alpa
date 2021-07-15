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
                   DeviceCluster, PhysicalDeviceMesh)
from parax.model.bert_model import BertConfig, FlaxBertForMaskedLMModule
from parax.model.gpt_model import FlaxGPTForLMModule

from parax.testing import assert_only_has_allreduce
from parax.util import run_cmd, write_tsv

import timeit

MB = 1024 ** 2
GB = 1024 ** 3


tic = time.time()
def log_time_stamp(message):
    global tic
    if message:
        print(f"{message}: {time.time() - tic:.2f} s")
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
    model_type = args.model
    batch_size, seq_len, hidden_size, num_layers, num_heads, vocab_size,\
        dp_size, tensor_mp_size = benchmark_case

    # Mesh configs
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh([dp_size, tensor_mp_size],
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
    def train_step(optimizer, batch, apply_func):
        def loss_func(params):
            rngs = {"dropout": batch["rng"]}
            logits = apply_func(params,
                                batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"],
                                batch["position_ids"],
                                rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Prepare model and input
    batch = {
        "input_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "token_type_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "position_ids": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "labels": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "rng": jax.random.PRNGKey(0),
    }
    log_time_stamp("Prepare input")

    # Init model and optimizer
    if model_type == "gpt":
        model = FlaxGPTForLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
        ))
    elif model_type == "bert":
        model = FlaxBertForMaskedLMModule(BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_layers,
            type_vocab_size=0,
        ))
    else:
        raise ValueError(f"Invalid model {model_type}")

    rngkey = jax.random.PRNGKey(0)
    params = model.init_dummy(rngkey, batch["input_ids"], batch["attention_mask"],
                              batch["token_type_ids"], batch["position_ids"])
    optimizer = optim.Adam(1e-2).create(params)
    params = rngkey = None
    log_time_stamp("Init model and optimizer")

    # Shard inputs and weights
    optimizer, batch = train_step.preshard_dynamic_args(optimizer, batch, model.apply)
    gc.collect()
    log_time_stamp("Compile and shard arguments")

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]

        optimizer = train_step(optimizer, batch, model.apply)
        physical_mesh.sync_workers()

        closure[0] = optimizer

    # Benchmark time cost
    func()
    stmt = "func()"
    repeat = 2
    number = args.number
    costs = np.array(timeit.repeat(stmt, globals={**globals(), **locals()},
        repeat=repeat, number=number)) / number
    real_mem = testing.last_compiled_executable.total_allocation_size()
    objective = testing.last_compiled_auto_sharding_objective

    # Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    print(f"#comm {hlo_ir.count('channel_id')}, " +
          f"#all-reduce {hlo_ir.count('all-reduce(') + hlo_ir.count('all-reduce-start(')}")
    #print(hlo_ir)

    #assert_only_has_allreduce(hlo_ir)
    #print("===== HLO =====")
    #print(hlo_ir)
    #optimizer = closure[0]
    #sharding_specs = jax.tree_util.tree_map(lambda x: x.sharding_spec, optimizer)

    # Log benchmark results
    heads = ["Type", "Case", "Mesh Shape", "Peak Mem", "Objective", "Mean Time", "Std Time"]
    values = [model_type, str(benchmark_case[:-2]), str(benchmark_case[-2:]),
             f"{real_mem/GB:.3f}", f"{objective:.2f}",
             f"{np.mean(costs):.3f}", f"{np.std(costs):.3f}"]
    write_tsv(heads, values, f"result_{model_type}.tsv")

    physical_mesh.shutdown()


# B = Batch size, S = seq_len, H = hidden size, L = num_layers,
# #head = num_heads, DP = dp_size, TMP = tensor_mp_size

benchmark_suite_4_gpu = [
    # B, S,    H,    L, #head,    V,     DP, TMP
    #(16, 1024, 1536, 6, 1536//96, 51200, 4,  1,),
    #(16, 1024, 1536, 6, 1536//96, 51200, 2,  2,),

    (4,  1024, 3072, 8,  3072//96, 51200, 4,  1),
    (4,  1024, 3072, 8,  3072//96, 51200, 2,  2),
]

benchmark_suite_8_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP
    #(32, 512,  1024, 22, 1024//64, 51200, 8,  1),
    (8,  1024, 3072, 8,  3072//96, 51200, 1,  8),
]

benchmark_suite_16_gpu = [
    # B, S,    H,    L,  #head,    V,     DP, TMP
    #(64, 512,  1024, 22, 1024//64, 51200, 16, 1),

    (16, 1024, 3072, 8,  3072//96, 51200, 2,  8),
]

def benchmark_all(use_profiling):
    num_gpus = ray.cluster_resources()["GPU"]
    benchmark_suites = {
        8 : benchmark_suite_8_gpu,
        16 : benchmark_suite_16_gpu,
    }

    for case in benchmark_suites[int(num_gpus)]:
        benchmark_transformer_one_case(case, use_profiling)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-profiling", action="store_true")
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--number", type=int, default=5)
    args = parser.parse_args()

    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    global_config.use_dummy_value_for_benchmarking = True

    benchmark_all(args.use_profiling)

