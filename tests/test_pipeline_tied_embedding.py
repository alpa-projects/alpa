import unittest
import os

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import ray

from parax import (parallelize, set_parallelize_options, mark_pipeline,
                   DeviceCluster, manual_layer_slicing)
from parax.testing import assert_allclose


class PipelineTiedEmbeddingTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address='auto')
        device_cluster = DeviceCluster()
        mesh = device_cluster.get_virtual_mesh()
        self.devices = mesh

    def tearDown(self):
        ray.shutdown()

    def train_tied_embedding(self, devices, strategy):
        vocab_size = 1024
        hidden_size = 16
        batch_size = 8
        seq_len = 8

        set_parallelize_options(devices=devices, strategy=strategy)

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

        def train_step(optimizer, x, y, apply_fn, use_manual_pipeline=False):

            def loss_func(params, x, y):
                out = apply_fn(params, x)
                y_ = jax.nn.one_hot(y, out.shape[-1])
                loss = -jnp.sum(y_ * jax.nn.log_softmax(out, axis=-1),
                                axis=-1).sum()
                mark_pipeline(name='2', mark_type='end')
                return loss

            if use_manual_pipeline:
                loss_func = manual_layer_slicing(loss_func)
            grad = jax.grad(loss_func)(optimizer.target, x, y)
            return grad

        x = jnp.ones((batch_size, seq_len), jnp.int32)
        y = jnp.ones((batch_size, seq_len), jnp.int32)

        # Init model and optimizer
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.Adam(1e-2).create(params)

        # Run and check results
        gradients = train_step(optimizer, x, y, model.apply)
        pipelined_train_step = parallelize(
            donate_argnums=())(lambda optimizer, x, y, apply_fn: train_step(
                optimizer, x, y, apply_fn, use_manual_pipeline=True))
        gradients_with_pipeline = pipelined_train_step(optimizer, x, y,
                                                       model.apply)
        assert_allclose(gradients, gradients_with_pipeline)
        pipelined_train_step.get_executable(optimizer, x, y,
                                            model.apply).shutdown()

    def test_tied_embedding_local_pipeline_parallel(self):
        self.train_tied_embedding(self.devices, "local_pipeline_parallel")

    def test_tied_embedding_3d_parallel(self):
        self.train_tied_embedding(self.devices, "3d_parallel")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(
        PipelineTiedEmbeddingTest(
            "test_tied_embedding_local_pipeline_parallel"))
    suite.addTest(PipelineTiedEmbeddingTest("test_tied_embedding_3d_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
