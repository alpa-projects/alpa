"""Test distributed weight initialization."""
import unittest

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax.tree_util import tree_flatten
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import optax

import alpa
from alpa import (init, shutdown, parallelize, ShardParallel, PipeshardParallel,
                  CreateStateParallel)


class CreateStateTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def run_test(self, method):
        use_bias = True
        batch_size = 8
        input_dim = output_dim = hidden_dim = 32

        grad_fn = (jax.grad if isinstance(method, ShardParallel) and
                   method.num_micro_batches is None else alpa.grad)

        class Model(nn.Module):

            @nn.compact
            def __call__(self, x):
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                if isinstance(method, PipeshardParallel):
                    alpa.mark_pipeline_boundary()
                x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
                x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
                return x

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                return jnp.mean((out - batch["y"])**2)

            grads = grad_fn(loss_func)(state.params)
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

        train_step = parallelize(train_step, method=method)
        create_state = parallelize(create_state,
                                   method=CreateStateParallel(
                                       train_step, batch))

        state = create_state()
        state = train_step(state, batch)

        if isinstance(method, ShardParallel):
            actual = tree_flatten(create_state.get_last_executable().
                                  get_output_placement_specs())[0]
            expected = tree_flatten(
                train_step.get_last_executable().get_input_placement_specs()
                [0])[0]
            assert actual == expected
        elif isinstance(method, PipeshardParallel):
            # The assertion is already in CreateStateExecutable::launch_on_driver
            # Here, we just call the function to test whether it is runnable.
            train_step.get_last_executable().get_output_placement_specs()

    def test_shard_parallel(self):
        method = ShardParallel(num_micro_batches=None)
        self.run_test(method)

    def test_shard_parallel_grad_acc(self):
        method = ShardParallel(num_micro_batches=2)
        self.run_test(method)

    def test_pipeshard_parallel(self):
        method = PipeshardParallel(num_micro_batches=2, layer_option="manual")
        self.run_test(method)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CreateStateTest("test_shard_parallel"))
    #suite.addTest(CreateStateTest("test_shard_parallel_grad_acc"))
    #suite.addTest(CreateStateTest("test_pipeshard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
