"""Start ray before the test"""
import jax
from jax import tree_flatten
import jax.numpy as jnp

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, manual_pipeline)
from parax.pipeline_parallel.primitive_def import mark_pipeline

from flax import linen as nn, optim

import ray
ray.init(address="auto")
jax.config.update('jax_platform_name', 'cpu')

class MLP_Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.output_dim, use_bias=False)(x)
        return x

batch_size = 128
hidden_dim = 2048
input_dim = output_dim = hidden_dim
model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
@manual_pipeline
def loss_func(params, x, y):
    out = model.apply(params, x)
    loss = jnp.mean((out - y)**2)
    mark_pipeline(name='2', mark_type='end')
    return loss
def train_step(optimizer, batch):
    param_grad, _x, _y = parax.grad(loss_func,
                                    argnums=(0, 1, 2))(optimizer.target,
                                                       batch['x'], batch['y'])
    new_optimizer = optimizer.apply_gradient(param_grad)
    return new_optimizer

x = jnp.ones((batch_size, input_dim))
y = jnp.ones((batch_size, output_dim))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)
batch = {'x': x, 'y': y}

virtual_mesh = DeviceCluster().get_virtual_mesh()
set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")
global_config.num_micro_batches = 1


parallel_train_step = parallelize(train_step, pipeline_marker_type='full')
new_optimizer = parallel_train_step(optimizer, batch)
targets = tree_flatten(new_optimizer.target)[0]

corr = tree_flatten(train_step(optimizer, batch).target)[0]
for tgt, cor in zip(targets, corr):
    assert jnp.allclose(tgt, cor)