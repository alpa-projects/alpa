import numpy as np
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp

from parax import parallelize, global_config, testing, SingleHostDeviceMesh
from parax.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection

import timeit

MB = 1024 ** 2


def block_until_ready(x):
    for leaf in jax.tree_util.tree_leaves(x):
        leaf.block_until_ready()


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
    num_devices = dp_size * tensor_mp_size
    device_mesh = SingleHostDeviceMesh(jax.devices()[:num_devices])
    logical_mesh = device_mesh.get_logical_mesh([dp_size, tensor_mp_size])

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
    optimizer, batch = train_step.preshard_dynamic_args(optimizer, batch, model.apply)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]

        optimizer = train_step(optimizer, batch, model.apply)
        block_until_ready(optimizer)

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

    # Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    objective = testing.last_compiled_auto_sharding_objective
    print("===== HLO =====")
    print(hlo_ir)

    #optimizer = closure[0]
    #sharding_specs = jax.tree_util.tree_map(lambda x: x.sharding_spec, optimizer)

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
    (16,          1024,    1536,        3,          1536//96,  2,       2),
    (16,          1024,    1536,        3,          1536//96,  1,       4),

    (8,           256,     2304,        3,          2304//96,  2,       2),
    (8,           256,     2304,        3,          2304//96,  1,       4),
]


def benchmark_all():
    for case in benchmark_suits:
        benchmark_transformer_one_case(case)


if __name__ == "__main__":
    benchmark_all()

