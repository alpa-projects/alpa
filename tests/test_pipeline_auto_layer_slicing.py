from functools import partial
import unittest

from flax import linen as nn, optim
import jax
import jax.numpy as jnp
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, automatic_layer_slicing)
from parax.model.bert_model import BertConfig
from parax.testing import MLPModel, TwoLayerBertLayerModel, assert_allclose, decorate_loss_fn


class PipelineAutoMarkerTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto")
        jax.config.update('jax_platform_name', 'cpu')

        virtual_mesh = DeviceCluster().get_virtual_mesh()
        set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")

    def tearDown(self):
        ray.shutdown()

    def test_mlp(self):
        batch_size = 256
        hidden_dim = 16
        input_dim = output_dim = hidden_dim

        # Init model
        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=False)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim))
        y = jax.random.normal(rngkey, (batch_size, output_dim))
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)
        batch = {'x': x, 'y': y}

        def train_step(optimizer, batch):

            def loss_func(params, x, y):
                out = model.apply(params, x)
                loss = jnp.mean((out - y)**2)
                return loss

            loss_func = decorate_loss_fn(loss_func, False, False, 2)

            param_grad = parax.grad(loss_func)(optimizer.target, batch['x'],
                                               batch['y'])
            new_optimizer = optimizer.apply_gradient(param_grad)
            return new_optimizer

        global_config.num_micro_batches = 4
        parallel_train_step = parallelize(train_step)

        # Run and check results
        expected = train_step(optimizer, batch).target
        actual = parallel_train_step(optimizer, batch).target
        assert_allclose(expected, actual)

        parallel_train_step.get_executable(optimizer, batch).shutdown()

    def test_2_layer_bert(self):

        def train_step(optimizer, batch, apply_fn):

            @partial(automatic_layer_slicing, layer_num=2, use_pipeline=True)
            def loss_func(params, x, y, attention_mask):
                out = apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                return loss

            grad_param = parax.grad(loss_func)(optimizer.target, batch['x'],
                                               batch['y'],
                                               batch['attention_mask'])

            new_optimizer = optimizer.apply_gradient(grad_param)
            return new_optimizer

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8

        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

        # Init model and optimizer
        model = TwoLayerBertLayerModel(config=BertConfig(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
            num_attention_heads=num_heads),
                                       manual_pipeline_layer=False)
        params = model.init(rngkey, x, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}

        global_config.num_micro_batches = 2
        pipelined_train_step = parallelize(train_step)

        # Run and check results
        expected = train_step(optimizer, batch, model.apply).target
        actual = pipelined_train_step(optimizer, batch, model.apply).target
        assert_allclose(expected, actual)
        pipelined_train_step.get_executable(optimizer, batch,
                                            model.apply).shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(PipelineAutoMarkerTest('test_mlp'))
    suite.addTest(PipelineAutoMarkerTest('test_2_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
