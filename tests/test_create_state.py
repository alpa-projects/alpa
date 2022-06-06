"""Test distributed weight initialization."""
import unittest

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import optax

import alpa
from alpa import init, parallelize, ShardParallel, CreateStateParallel


class CreateStateTest(unittest.TestCase):
    def setUp(self):
        init(cluster="ray")

    def test_shard_parallel(self):
        use_bias = False
        batch_size = 8
        input_dim = output_dim = hidden_dim = 128

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
                return x

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = alpa.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        def create_state():
            model = Model()
            rngkey = jax.random.PRNGKey(0)
            params = model.init(rngkey, jnp.ones((1, input_dim)))
            tx = optax.adam(learning_rate=1e-2)
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        batch = {
            "x": jnp.ones((batch_size, input_dim)),
            "y": jnp.ones((batch_size, output_dim)),
        }

        # Test with and without gradient accumulation
        for num_micro_batches in [None, 2]:
            p_train_step = parallelize(train_step,
                                     method=ShardParallel(num_micro_batches=num_micro_batches))
            p_create_state = parallelize(create_state, method=CreateStateParallel(p_train_step, batch))

            state = p_create_state()
            state = p_train_step(state, batch)

            actual = p_create_state.get_last_executable().output_sharding_specs

            if num_micro_batches == None:
                # Test NormalMeshDriverExecutable
                expected = p_train_step.get_last_executable().input_sharding_specs[:len(actual)]
            else:
                # Test GradAccMeshDriverExecutable
                expected = p_train_step.get_last_executable().global_arg_sharding_specs[:len(actual)]

            for x, y in zip(actual, expected):
                assert x == y


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CreateStateTest("test_shard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

