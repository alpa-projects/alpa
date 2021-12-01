import unittest
import os

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import ray

from parax import (parallelize, set_parallelize_options, mark_pipeline,
                   DeviceCluster, manual_layer_slicing)
from parax.testing import BertLayerModel, assert_allclose
from parax.model.bert_model import BertConfig, FlaxBertLayer


class PipelineBERTTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        assert len(jax.local_devices()) >= 4

        ray.init(address='auto')
        device_cluster = DeviceCluster()
        mesh = device_cluster.get_virtual_physical_mesh()
        self.devices = mesh

    def tearDown(self):
        ray.shutdown()

    def train_2_layer_bert(self, devices, strategy):
        set_parallelize_options(devices=devices, strategy=strategy)

        def train_step(optimizer, batch, apply_fn):

            def loss_func(params, x, y, attention_mask):
                out = apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name='2', mark_type='end')
                return loss

            loss_func = manual_layer_slicing(loss_func)

            grad_param = jax.grad(loss_func)(optimizer.target, batch['x'],
                                             batch['y'],
                                             batch['attention_mask'])

            # new_optimizer = optimizer.apply_gradient(grad_param)
            return grad_param

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8
        dtype = jnp.float32

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=dtype)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=dtype)

        # Init model and optimizer
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=2))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)

        # Train step
        gradients = train_step(optimizer, {
            "x": x,
            "y": y,
            "attention_mask": attention_mask
        }, model.apply)
        pipelined_train_step = parallelize(
            donate_argnums=())(lambda optimizer, batch, apply_fn: train_step(
                optimizer, batch, apply_fn))
        args = (optimizer, {
            "x": x,
            "y": y,
            "attention_mask": attention_mask
        }, model.apply)
        executable = pipelined_train_step.get_executable(*args)
        gradients_with_pipeline = pipelined_train_step(*args)

        # Check results
        assert_allclose(gradients, gradients_with_pipeline)
        executable.shutdown()

    def test_2_layer_bert_local_pipeline_parallel(self):
        self.train_2_layer_bert(self.devices, "local_pipeline_parallel")

    def test_2_layer_bert_3d_parallel(self):
        self.train_2_layer_bert(self.devices, "3d_parallel")


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineBERTTest("test_2_layer_bert_local_pipeline_parallel"))
    suite.addTest(PipelineBERTTest("test_2_layer_bert_3d_parallel"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
