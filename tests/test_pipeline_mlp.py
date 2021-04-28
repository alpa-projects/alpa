from jax.interpreters.ad import linear_transpose2
import numpy as np

import unittest

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis
from flax import linen as nn
from flax import optim

from parax import parallelize, global_config, testing, mark_pipeline

from test_auto_sharding_basic import assert_close, all_reduce_cost

MB = 1024 ** 2

class AutoShardingMLPTest(unittest.TestCase):
    def setUp(self):
        assert len(jax.local_devices()) >= 4
        self.devices = tuple(jax.local_devices()[:4])

    def test_2_layer_mlp(self):
        class Model(nn.Module):
            hidden_dim: int
            output_dim: int

            @nn.compact
            def __call__(self, x):
                x, = mark_pipeline(x, name='1', mark_type='start')
                x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
                x = nn.relu(x)
                x, = mark_pipeline(x, name='1', mark_type='end')
                x, = mark_pipeline(x, name='2', mark_type='start')
                x = nn.Dense(features=self.output_dim, use_bias=False)(x)
                return x

        @parallelize(memory_budget_per_device=30 * (1 << 20),
                     devices=self.devices)
        def train_step(optimizer, batch, apply_fn):
            def loss_func(params):
                out = apply_fn(params, batch['x'])
                loss = jnp.mean((out - batch['y']) ** 2)
                loss, = mark_pipeline(loss, name='2', mark_type='end')
                return loss

            grad = jax.grad(loss_func)(optimizer.target)
            new_optimizer = optimizer.apply_gradient(grad)
            return new_optimizer

        batch_size = 128
        hidden_dim = 2048
        input_dim = output_dim = hidden_dim

        x = jnp.ones((batch_size, input_dim))
        y = jnp.ones((batch_size, output_dim))

        # Init model and optimizer
        model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # JIT compiler
        optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AutoShardingMLPTest('test_2_layer_mlp'))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

