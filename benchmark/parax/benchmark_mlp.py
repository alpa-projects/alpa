import numpy as np
from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp

from parax import parallelize, global_config, testing, SingleHostDeviceMesh

from timeit import timeit

MB = 1024 ** 2


def replicate(a, devices):
    a = jax.pmap(lambda x, y: x, in_axes=(None, 0), out_axes=None, devices=devices)\
                (a, jnp.ones(len(devices)))
    return a


def block_until_ready(x):
    jax.tree_util.tree_leaves(x)[-1].block_until_ready()


def compute_bytes(param_tree):
    n_bytes = 4
    param_tree = jax.tree_util.tree_map(lambda arr: np.prod(arr.shape) * n_bytes,
                                        param_tree)
    total = np.sum(jax.tree_util.tree_flatten(param_tree)[0])
    return total


def benchmark_mlp():
    class Model(nn.Module):
        hidden_size: int
        num_layers: int

        @nn.compact
        def __call__(self, x):
            for i in range(self.num_layers):
                x = nn.Dense(features=self.hidden_dim * 4)(x)
                x = nn.gelu(x)
                x = nn.Dense(features=self.hidden_dim)(x)
            return x

    device_mesh = SingleHostDeviceMesh(jax.devices()[:4])

    @parallelize(devices=device_mesh)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Model configs
    batch_size = 16
    hidden_size = 2304
    seq_length = 1024
    num_layers = 4
    num_attention_heads = hidden_size // 96

    # Prepare input
    x = jnp.ones((batch_size, seq_length, hidden_size))
    y = jnp.ones((batch_size, seq_length, hidden_size))
    x = device_mesh.put_replicated(x)
    y = device_mesh.put_replicated(y)

    print(x.sharding_spec)
    exit()

    # Initialize model
    model = Model(hidden_dim=hidden_dim)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, x)
    optimizer = optim.SGD(1e-2).create(params)
    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]
        block_until_ready(optimizer)
        optimizer = train_step(optimizer,
                               {"x": x, "y": y},
                               model.apply)
        block_until_ready(optimizer)
        closure[0] = optimizer

    # Benchmark time cost
    func()
    func()
    stmt = "func()"
    repeat = 2
    number = 4
    costs = np.array(timeit.repeat(stmt, globals={**globals(), **locals()},
        repeat=repeat, number=number)) / number
    real_mem = testing.last_compiled_executable.total_allocation_size()

    ## Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    objective = testing.last_compiled_auto_sharding_objective

    optimizer = closure[0]
    sharding_specs = jax.tree_util.tree_map(lambda x: x.sharding_spec, optimizer)
    #print(hlo_ir)
    #print(sharding_specs)

    return real_mem, cost, objective


if __name__ == '__main__':
    benchmark_mlp()

