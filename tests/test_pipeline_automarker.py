import copy
from functools import partial
import unittest

from flax import linen as nn, optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, automatic_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertLayer
from parax.testing import assert_allclose


class MLP_Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.Dense(features=self.output_dim, use_bias=True)(x)
        return x


class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        layer_outputs = self.layer0(x, attention_mask)
        x = layer_outputs[0]
        layer_outputs = self.layer1(x, attention_mask)
        x = layer_outputs[0]
        return x


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
        model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
        x = jnp.array(np.random.rand(batch_size, input_dim))
        y = jnp.array(np.random.rand(batch_size, output_dim))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x)
        optimizer = optim.GradientDescent(1e-2).create(params)
        batch = {'x': x, 'y': y}

        @partial(automatic_layer_slicing, layer_num=2, use_pipeline=True)
        def loss_func(params, x, y):
            out = model.apply(params, x)
            loss = jnp.mean((out - y)**2)
            return loss

        def train_step(optimizer, batch):
            param_grad = parax.grad(loss_func)(optimizer.target, batch['x'],
                                               batch['y'])
            new_optimizer = optimizer.apply_gradient(param_grad)
            return new_optimizer.target

        global_config.num_micro_batches = 4

        # copy to prevent from donation
        corr = train_step(optimizer, batch)
        parallel_train_step = parallelize(train_step)
        new_optimizer = parallel_train_step(optimizer, batch)
        assert_allclose(new_optimizer, corr)
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
            return new_optimizer.target

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8

        x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        y = jnp.ones(
            (batch_size, seq_len, hidden_size),
            dtype=jnp.float32) * 23  # * np.arange(hidden_size)[None, None, :]
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

        # Init model and optimizer
        model = BertLayer_Model(
            config=BertConfig(hidden_size=hidden_size,
                              intermediate_size=hidden_size * 4,
                              num_attention_heads=num_heads))
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, x, attention_mask)
        optimizer = optim.GradientDescent(1e-2).create(params)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}

        global_config.num_micro_batches = 2

        corr_tgt = train_step(optimizer, batch, model.apply)
        pipelined_train_step = parallelize(train_step)
        pipe_tgt = pipelined_train_step(optimizer, batch, model.apply)
        assert_allclose(corr_tgt, pipe_tgt)
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
