import unittest

import jax
from jax import tree_flatten
import jax.numpy as jnp
import numpy as np
from parax.testing import assert_allclose
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, manual_pipeline)
from parax.pipeline_parallel.primitive_def import mark_pipeline

from flax import linen as nn, optim


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


class AccumulateGradTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')
        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")

    def tearDown(self):
        ray.shutdown()

    def test_mlp(self):
        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim
        model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
        x = jnp.array(np.random.rand(batch_size, input_dim))
        y = jnp.array(np.random.rand(batch_size, output_dim))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)
        batch = {'x': x, 'y': y}

        @manual_pipeline
        def loss_func(params, x, y):
            out = model.apply(params, x)
            loss = jnp.mean((out - y)**2)
            mark_pipeline(name='2', mark_type='end')
            return loss

        def train_step(optimizer, batch):
            param_grad, _x, _y = parax.grad(loss_func,
                                            argnums=(0, 1, 2))(optimizer.target,
                                                               batch['x'],
                                                               batch['y'])
            new_optimizer = optimizer.apply_gradient(param_grad)
            return new_optimizer

        global_config.num_micro_batches = 4

        parallel_train_step = parallelize(train_step)
        new_optimizer = parallel_train_step(optimizer, batch)
        targets = tree_flatten(new_optimizer.target)[0]

        corr = tree_flatten(train_step(optimizer, batch).target)[0]
        assert_allclose(targets, corr)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())