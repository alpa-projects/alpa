"""Test following another parallel strategy."""
import unittest

from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

import alpa
from alpa import init, shutdown, parallelize, ShardParallel, PipeshardParallel


class FollowParallelTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def run_test(self, method):
        use_bias = True
        batch_size = 32
        input_dim = output_dim = hidden_dim = 8

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

            grads = grad_fn(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state

        def eval_step(params, batch):
            out = state.apply_fn(params, batch["x"])
            return jnp.mean((out - batch["y"])**2)

        def create_state():
            model = Model()
            rngkey = jax.random.PRNGKey(0)
            params = model.init(rngkey, jnp.ones((1, input_dim)))
            tx = optax.adam(learning_rate=1e-2)
            return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        train_batch = {
            "x": jnp.ones((batch_size, input_dim)),
            "y": jnp.ones((batch_size, output_dim)),
        }
        eval_batch = {
            "x": jnp.ones((batch_size * 2, input_dim)),
            "y": jnp.ones((batch_size * 2, output_dim)),
        }

        grad_fn = jax.grad if method.num_micro_batches is None else alpa.grad
        num_micro_batches = method.num_micro_batches

        state = create_state()

        train_step = parallelize(train_step, method=method)
        eval_step = parallelize(eval_step,
                                method=alpa.FollowParallel(
                                    train_step,
                                    num_micro_batches=num_micro_batches))

        state = train_step(state, train_batch)
        out = eval_step(state.params, eval_batch)

        actual = jax.tree_flatten(
            eval_step.get_last_executable().get_input_placement_specs()[0])[0]
        expected = jax.tree_flatten(
            train_step.get_last_executable().get_input_placement_specs()
            [0].params)[0]
        assert actual == expected

    def test_shard_parallel(self):
        method = ShardParallel(num_micro_batches=None)
        self.run_test(method)

    def test_shard_parallel_grad_acc(self):
        method = ShardParallel(num_micro_batches=2)
        self.run_test(method)

    def test_pipeshard_parallel(self):
        method = PipeshardParallel(
            num_micro_batches=2, layer_option=alpa.AutoLayerOption(layer_num=2))
        self.run_test(method)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FollowParallelTest("test_shard_parallel"))
    suite.addTest(FollowParallelTest("test_shard_parallel_grad_acc"))
    suite.addTest(FollowParallelTest("test_pipeshard_parallel"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
