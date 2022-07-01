import unittest
import os

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from alpa import (init, parallelize, mark_pipeline_boundary, grad,
                  PipeshardParallel)
from alpa.model.model_util import TrainState
from alpa.testing import assert_allclose


class PipelineTiedEmbeddingTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def train_tied_embedding(self, method):
        vocab_size = 256
        hidden_size = 16
        batch_size = 8
        seq_len = 8

        class Model(nn.Module):
            """Tied input and output embedding."""

            def setup(self):
                self.embed = nn.Embed(vocab_size, hidden_size)

            def __call__(self, x):
                x = self.embed(x)
                mark_pipeline_boundary()
                embed = self.embed.variables["params"]["embedding"]
                x = x @ embed.T
                return x

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                y_ = jax.nn.one_hot(batch["y"], out.shape[-1])
                loss = -jnp.sum(y_ * jax.nn.log_softmax(out, axis=-1),
                                axis=-1).sum()
                return loss

            grads = grad(loss_func)(state.params)
            return state.apply_gradients(grads=grads)

        x = jnp.ones((batch_size, seq_len), jnp.int32)
        y = jnp.ones((batch_size, seq_len), jnp.int32)

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        tx = optax.adam(learning_rate=1e-2)
        state = TrainState.create(apply_fn=model.apply,
                                  params=params,
                                  tx=tx,
                                  dynamic_scale=None)

        # Run and check results
        p_train_step = parallelize(train_step, method=method)
        batch = {"x": x, "y": y}
        expected_new_state = train_step(state, batch)
        actual_new_state = p_train_step(state, batch)
        assert_allclose(actual_new_state.params, expected_new_state.params)

    def test_tied_embedding_pipeshard_parallel(self):
        method = PipeshardParallel(num_micro_batches=2, layer_option="manual")
        self.train_tied_embedding(method)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        PipelineTiedEmbeddingTest("test_tied_embedding_pipeshard_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
