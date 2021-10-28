import copy
import unittest

from flax import linen as nn
import optax
import jax
import jax.numpy as jnp
import numpy as np
import ray

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, manual_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertLayer, TrainState
from parax.pipeline_parallel.primitive_def import mark_pipeline
from parax.testing import assert_allclose


def create_train_state(rngkey, model, params):
    params = model.init(rngkey, *params)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


class MLPModel(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.Dense(features=self.output_dim, use_bias=True)(x)
        return x


class BertLayerModel(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        mark_pipeline(name='1', mark_type='start')
        out = x
        out = self.layer0(out, attention_mask)[0]
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        out = self.layer1(out, attention_mask)[0]
        return out


class AccumulateGradTest(unittest.TestCase):

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

        model = MLPModel(hidden_dim=hidden_dim, output_dim=output_dim)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim))
        y = jax.random.normal(rngkey, (batch_size, output_dim))
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        def train_step(state, batch):

            @manual_layer_slicing
            def loss_func(params):
                out = state.apply_fn(params, batch["x"])
                loss = jnp.mean((out - batch["y"])**2)
                mark_pipeline(name='2', mark_type='end')
                return loss

            param_grad = parax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=param_grad)
            return new_state

        global_config.num_micro_batches = 4

        nstep = 3
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch)
        expected_new_state = None
        actual_new_state = None
        for i in range(nstep):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch)
            assert_allclose(expected_new_state.params, actual_new_state.params)

        executable.shutdown()

    def test_2_layer_bert(self):

        def train_step(state, batch, apply_fn):

            @manual_layer_slicing
            def loss_func(params):
                out = state.apply_fn(params, batch["x"],
                                     batch["attention_mask"])
                loss = jnp.mean((out - batch["y"])**2)
                mark_pipeline(name='2', mark_type='end')
                return loss

            grad_param = parax.grad(loss_func)(state.params)
            new_state = state.apply_gradients(grads=grad_param)
            return new_state

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
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads))
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch,
                                                        model.apply)
        expected_new_state = None
        actual_new_state = None

        for i in range(3):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch, model.apply)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch, model.apply)
            assert_allclose(expected_new_state.params, actual_new_state.params,
                            5e-4, 5e-4)

        executable.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    suite.addTest(AccumulateGradTest('test_2_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
