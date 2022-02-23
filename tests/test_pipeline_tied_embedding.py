import unittest
import os

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax
import ray

from alpa import (parallelize, set_parallelize_options, mark_pipeline,
                  DeviceCluster, manual_layer_construction, grad)
from alpa.model.model_util import TrainState
from alpa.testing import assert_allclose
from alpa.util import get_ray_namespace_str


class PipelineTiedEmbeddingTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address="auto",
                 namespace=get_ray_namespace_str(prefix="alpa-unittest"))
        device_cluster = DeviceCluster()
        self.devices = device_cluster.get_virtual_physical_mesh()

    def tearDown(self):
        ray.shutdown()

    def train_tied_embedding(self, devices, strategy, num_micro_batches):
        vocab_size = 1024
        hidden_size = 16
        batch_size = 8
        seq_len = 8

        set_parallelize_options(devices=devices,
                                strategy=strategy,
                                num_micro_batches=num_micro_batches)

        class Model(nn.Module):
            """Tied input and output embedding."""

            def setup(self):
                self.embed = nn.Embed(vocab_size, hidden_size)

            def __call__(self, x):
                mark_pipeline(name='1', mark_type='start')
                x = self.embed(x)
                mark_pipeline(name='1', mark_type='end')
                mark_pipeline(name='2', mark_type='start')
                embed = self.embed.variables["params"]["embedding"]
                x = x @ embed.T
                return x

        def train_step(state, batch):

            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                y_ = jax.nn.one_hot(batch["y"], out.shape[-1])
                loss = -jnp.sum(y_ * jax.nn.log_softmax(out, axis=-1),
                                axis=-1).sum()
                mark_pipeline(name='2', mark_type='end')
                return loss

            loss_func = manual_layer_construction(loss_func)
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
        pipelined_train_step = parallelize(train_step)
        batch = {"x": x, "y": y}
        expected_new_state = train_step(state, batch)
        actual_new_state = pipelined_train_step(state, batch)
        assert_allclose(actual_new_state.params, expected_new_state.params)

        pipelined_train_step.get_executable(state, batch).shutdown()

    def test_tied_embedding_pipeshard_parallel(self):
        self.train_tied_embedding(self.devices, "pipeshard_parallel", 2)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        PipelineTiedEmbeddingTest("test_tied_embedding_pipeshard_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
