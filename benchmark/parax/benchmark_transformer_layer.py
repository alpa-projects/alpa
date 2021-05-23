import gc
import os
import time

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

from parax import parallelize, global_config, testing, DeviceCluster
from parax.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection
from parax.util import run_cmd

import timeit

MB = 1024 ** 2


def compute_bytes(param_tree):
    n_bytes = 4
    param_tree = jax.tree_util.tree_map(lambda arr: np.prod(arr.shape) * n_bytes,
                                        param_tree)
    total = np.sum(jax.tree_util.tree_flatten(param_tree)[0])
    return total


def benchmark_transformer_one_case(benchmark_case):
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, dp_size, tensor_mp_size =\
        benchmark_case

    # Mesh configs
    device_cluster = DeviceCluster()
    physical_mesh = device_cluster.get_physical_mesh()
    logical_mesh = physical_mesh.get_logical_mesh(
        [dp_size, tensor_mp_size], [1, 1], [1, 0.01])

    @parallelize(devices=logical_mesh)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch["hidden_states"], batch["attention_mask"])[0]
            return jnp.mean((out - batch["label"]) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Prepare model and input
    batch = {
        "hidden_states": jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "label": jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32),
    }

    # Init model and optimizer
    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, batch["hidden_states"], batch["attention_mask"])
    optimizer = optim.Adam(1e-2).create(params)
    params = rngkey = None
    optimizer, batch = train_step.preshard_dynamic_args(optimizer, batch, model.apply)
    gc.collect()

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]

        optimizer = train_step(optimizer, batch, model.apply)
        physical_mesh.sync_workers()

        closure[0] = optimizer

    # Benchmark time cost
    func()
    func()
    stmt = "func()"
    repeat = 2
    number = 10
    costs = np.array(timeit.repeat(stmt, globals={**globals(), **locals()},
        repeat=repeat, number=number)) / number
    real_mem = testing.last_compiled_executable.total_allocation_size()
    objective = testing.last_compiled_auto_sharding_objective

    # Check sharding strategy
    #hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    #hlo_ir = hlo_module.to_string()
    #print("===== HLO =====")
    #print(hlo_ir)
    #optimizer = closure[0]
    #sharding_specs = jax.tree_util.tree_map(lambda x: x.sharding_spec, optimizer)

    # Log benchmark results
    heads = ["Type", "Case", "Mesh Shape", "Peak Mem", "Mean Time", "Std Time", "Objective"]
    values = ["transformer-layer", str(benchmark_case[:-2]), str(benchmark_case[-2:]),
             f"{real_mem/MB:.2f}", f"{np.mean(costs):.2f}", f"{np.std(costs):.2f}",
             f"{objective:.2f}"]

    line = ""
    for i in range(len(heads)):
        line += heads[i] + ": " + values[i] + "  "
    print(line)

    with open("results.tsv", "a") as fout:
        fout.write("\t".join(values) + "\n")


benchmark_suits = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size
    (32,          1024,    1536,        1,          1536//96,  4,       1),
    (32,          1024,    1536,        1,          1536//96,  4,       2),
    (32,          1024,    1536,        1,          1536//96,  2,       4),
    (32,          1024,    1536,        1,          1536//96,  1,       8),

    (32,          128,     5120,        1,          5120//128, 8,       1),
    (32,          128,     5120,        1,          5120//128, 4,       2),
    (32,          128,     5120,        1,          5120//128, 2,       4),
    (32,          128,     5120,        1,          5120//128, 1,       8),
]


def benchmark_all():
    for case in benchmark_suits:
        benchmark_transformer_one_case(case)
        break


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    ray.init(address="auto")
    benchmark_all()

