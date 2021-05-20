"""Test distributed mulit-host device mesh."""

import jax
import jax.numpy as jnp
import numpy as np
import os
from flax import linen as nn
from flax import optim

from parax import parallelize, DeviceCluster, global_config
from parax.testing import assert_allclose

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
global_config.shard_parallel_strategy = "auto_sharding"
device_cluster = DeviceCluster()

physical_mesh = device_cluster.get_physical_mesh()


class Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

def train_step(optimizer, batch, apply_fn):
    def loss_func(params):
        out = apply_fn(params, batch['x'])
        return jnp.mean((out - batch['y']) ** 2)

    grad = jax.grad(loss_func)(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grad)
    return new_optimizer

batch_size = 512
input_dim = hidden_dim = output_dim = 32

# One batch of data and label
batch = {
    "x": np.random.randn(batch_size, input_dim),
    "y": np.random.randn(batch_size, output_dim),
}

# Init model and optimizer
model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, batch['x'])
optimizer = optim.GradientDescent(1e-2).create(params)

# Serial execution
optimizer_expected = train_step(optimizer, batch, model.apply)

# Distributed execution
train_step_parallel = parallelize(devices=physical_mesh)(train_step)
optimizer_actual = train_step_parallel(optimizer, batch, model.apply)

# Check results
assert_allclose(optimizer_expected.target, optimizer_actual.target)
