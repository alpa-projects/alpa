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
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        dynamic_scale=None)
    return state


class MLP_Model(nn.Module):
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


class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        mark_pipeline(name='1', mark_type='start')
        layer_outputs = self.layer0(x, attention_mask)
        x = layer_outputs[0]
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        layer_outputs = self.layer1(x, attention_mask)
        x = layer_outputs[0]
        return x


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
        model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
        x = jnp.array(np.random.rand(batch_size, input_dim))
        y = jnp.array(np.random.rand(batch_size, output_dim))
        rngkey = jax.random.PRNGKey(0)
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        @manual_layer_slicing
        def loss_func(params, x, y):
            out = model.apply(params, x)
            loss = jnp.mean((out - y)**2)
            mark_pipeline(name='2', mark_type='end')
            return loss

        def train_step(state, batch):
            param_grad = parax.grad(loss_func)(state.params, batch['x'],
                                               batch['y'])
            new_state = state.apply_gradients(grads=param_grad)
            return new_state

        global_config.num_micro_batches = 4

        nstep = 5
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch)
        expected_new_state = None
        actual_new_state = None
        # Test ReplicatedDistributedArray correctness
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
            def loss_func(params, x, y, attention_mask):
                out = apply_fn(params, x, attention_mask)
                loss = jnp.mean((out - y)**2)
                mark_pipeline(name='2', mark_type='end')
                return loss

            grad_param = parax.grad(loss_func)(state.params, batch['x'],
                                               batch['y'],
                                               batch['attention_mask'])

            new_state = state.apply_gradients(grads=grad_param)
            return new_state

        batch_size = 16
        seq_len = 8
        hidden_size = 512
        num_heads = 8
        x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
        y = jnp.ones((batch_size, seq_len, hidden_size),
                     dtype=jnp.float32) * 23  # * np.arange(hidden_size)[None, None, :]
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        # x = jnp.array(np.random.rand(batch_size, seq_len, hidden_size), dtype=jnp.float32)
        # y = jnp.array(np.random.rand(batch_size, seq_len, hidden_size), dtype=jnp.float32) * 23.0
        # attention_mask = jnp.array(np.random.rand(batch_size, seq_len), dtype=jnp.float32)

        # Init model and optimizer
        model = BertLayer_Model(
            config=BertConfig(hidden_size=hidden_size,
                              intermediate_size=hidden_size * 4,
                              num_attention_heads=num_heads))
        rngkey = jax.random.PRNGKey(0)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        global_config.num_micro_batches = 2
        parallel_train_step = parallelize(train_step)
        executable = parallel_train_step.get_executable(state, batch, model.apply)
        expected_new_state = None
        actual_new_state = None

        # Test ReplicatedDistributedArray correctness
        for i in range(5):
            if i > 0:
                state = expected_new_state
            expected_new_state = train_step(state, batch, model.apply)
            if i > 0:
                state = actual_new_state
            actual_new_state = parallel_train_step(state, batch, model.apply)
            assert_allclose(expected_new_state.params, actual_new_state.params)

        executable.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(AccumulateGradTest('test_mlp'))
    # TODO(Hao): disable this test and move it back when the precision issue is fixed.
    # suite.addTest(AccumulateGradTest('test_2_layer_bert'))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
