import numpy as np
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp

from parax import parallelize, global_config, testing, SingleHostDeviceMesh

import timeit

MB = 1024 ** 2


def block_until_ready(x):
    jax.tree_util.tree_leaves(x)[-1].block_until_ready()


def compute_bytes(param_tree):
    n_bytes = 4
    param_tree = jax.tree_util.tree_map(lambda arr: np.prod(arr.shape) * n_bytes,
                                        param_tree)
    total = np.sum(jax.tree_util.tree_flatten(param_tree)[0])
    return total


def benchmark_mlp_one_case(benchmark_case):
    # Model configs
    batch_size, seq_len, hidden_size, num_layers, num_heads, dp_size, tensor_mp_size =\
        benchmark_case

    class Model(nn.Module):
        hidden_size: int
        num_layers: int

        @nn.compact
        def __call__(self, x):
            for i in range(self.num_layers):
                x = nn.Dense(features=self.hidden_size * 4, use_bias=False)(x)
                x = nn.gelu(x)
                x = nn.Dense(features=self.hidden_size, use_bias=False)(x)
            return x

    # Mesh configs
    num_devices = dp_size * tensor_mp_size
    device_mesh = SingleHostDeviceMesh(jax.devices()[:num_devices])
    logical_mesh = device_mesh.get_logical_mesh([dp_size, tensor_mp_size])

    @parallelize(devices=logical_mesh)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Prepare model and input
    batch = {
        "x": jnp.ones((batch_size, seq_len, hidden_size)),
        "y": jnp.ones((batch_size, seq_len, hidden_size)),
    }
    model = Model(hidden_size=hidden_size, num_layers=num_layers)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, batch["x"])
    optimizer = optim.GradientDescent(1e-2).create(params)
    optimizer, batch = train_step.preshard_dynamic_args(optimizer, batch, model.apply)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]

        block_until_ready(optimizer)
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

    line = f"Case: {benchmark_case}\t"\
           f"PeakMem: {real_mem/MB:.2f}\t"\
           f"Mean Time: {np.mean(costs):.2f}\t"\
           f"Std Time: {np.std(costs):.2f}\t"\
           f"Objective: {objective:.2f}\t"

    print(line)
    with open("results.tsv", "a") as fout:
        fout.write(line + "\n")


benchmark_suits = [
    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size,
    (16,          1024,    2304,        4,          2304//96,  4,       1),
    (16,          1024,    2304,        4,          2304//96,  2,       2),
    (16,          1024,    2304,        4,          2304//96,  1,       4),

    # Batch size, seq_len, hidden size, num_layers, num_heads, dp_size, tensor_mp_size,
    (8,           256,     2304,        4,          2304//96,  4,       1),
    (8,           256,     2304,        4,          2304//96,  2,       2),
    (8,           256,     2304,        4,          2304//96,  1,       4),
]


def benchmark_mlp():
    for case in benchmark_suits:
        benchmark_mlp_one_case(case)


if __name__ == "__main__":
    benchmark_mlp()

