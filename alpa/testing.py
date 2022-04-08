"""Utilities for testing."""
from collections.abc import Iterable
import time
import unittest

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax
import jax
from jax.experimental.maps import FrozenDict as FrozenDictJax
import jax.numpy as jnp
import numpy as np
import optax
import ray

from alpa.api import parallelize, grad, value_and_grad
from alpa.device_mesh import DeviceCluster
from alpa.global_env import set_parallelize_options, global_config
from alpa.model.bert_model import BertConfig, FlaxBertLayer
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.layer_construction import (
    automatic_layer_construction, manual_layer_construction)
from alpa.pipeline_parallel.primitive_def import mark_pipeline
from alpa.util import get_ray_namespace_str


def assert_allclose(x, y, rtol=1e-4, atol=1e-4):
    """Assert the arrays in x and y are all close."""
    if isinstance(x, (dict, FrozenDictJax, FrozenDictFlax)):
        assert isinstance(y, (dict, FrozenDictJax, FrozenDictFlax))
        assert set(x.keys()) == set(y.keys())
        for k in x.keys():
            assert_allclose(x[k], y[k], rtol, atol)
    elif isinstance(x, Iterable) and not hasattr(x, '__array__'):
        assert isinstance(y, Iterable) and not hasattr(y, '__array__')
        assert len(x) == len(y)
        for x_elt, y_elt in zip(x, y):
            assert_allclose(x_elt, y_elt, rtol, atol)
    elif hasattr(x, '__array__') or np.isscalar(x):
        assert hasattr(y, '__array__') or np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        np.testing.assert_allclose(x, y, rtol, atol)
    elif x == y:
        return
    else:
        raise TypeError((type(x), type(y)))


# Models and functions for Pipeline Tests
class MLPModel(nn.Module):
    hidden_dim: int
    output_dim: int
    manual_pipeline_layer: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(self, x):
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='end')
            mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        x = nn.Dense(features=self.output_dim, use_bias=self.use_bias)(x)
        return x


class BertLayerModel(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    manual_pipeline_layer: bool = True

    def setup(self):
        self.layers = [
            FlaxBertLayer(config=self.config, dtype=self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]

    def __call__(self, x, attention_mask):
        for i, layer in enumerate(self.layers):
            if self.manual_pipeline_layer:
                mark_pipeline(name=str(i), mark_type='start')
            layer_outputs = layer(x, attention_mask)
            x = layer_outputs[0]
            if self.manual_pipeline_layer and i != len(self.layers) - 1:
                mark_pipeline(name=str(i), mark_type='end')
        return x


def create_train_state(rngkey, model, inputs):
    params = model.init(rngkey, *inputs)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


def create_dummy_train_state(rngkey, model, inputs, dtype=jnp.float16):
    params = model.init_dummy(rngkey, *inputs)
    tx = optax.adam(learning_rate=1e-2)
    mixed_precision = (dtype == jnp.float16)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              mixed_precision=mixed_precision,
                              dynamic_scale=None)
    return state


def decorate_loss_fn(fn, manual_pipeline, use_remat, layer_num):
    if manual_pipeline:
        return manual_layer_construction(fn, remat_layer=use_remat)
    return automatic_layer_construction(fn,
                                        remat_layer=use_remat,
                                        layer_num=layer_num)


def get_mlp_train_step(use_parallel,
                       manual_pipeline_layer,
                       test_remat,
                       return_value=False):

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            if manual_pipeline_layer:
                mark_pipeline(name='2', mark_type='end')
            return loss

        if use_parallel:
            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         test_remat, 2)
            if return_value:
                val, grads = value_and_grad(loss_func)(state.params)
            else:
                val, grads = 0, grad(loss_func)(state.params)
        else:
            if return_value:
                val, grads = jax.value_and_grad(loss_func)(state.params)
            else:
                val, grads = 0, jax.grad(loss_func)(state.params)

        new_state = state.apply_gradients(grads=grads)
        if return_value:
            return new_state, val
        return new_state

    if use_parallel:
        return parallelize(train_step)
    else:
        return train_step


def get_bert_layer_train_step(use_parallel,
                              manual_pipeline_layer,
                              test_remat,
                              num_layers,
                              decorate=None,
                              return_value=False):
    if decorate is None:
        decorate = use_parallel

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"], batch["attention_mask"])
            loss = jnp.mean((out - batch["y"])**2)
            if manual_pipeline_layer:
                mark_pipeline(name=str(num_layers - 1), mark_type='end')
            return loss

        if decorate:
            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         test_remat, num_layers)
            if return_value:
                val, grads = value_and_grad(loss_func)(state.params)
            else:
                val, grads = 0, grad(loss_func)(state.params)
        else:
            if return_value:
                val, grads = jax.value_and_grad(loss_func)(state.params)
            else:
                val, grads = 0, jax.grad(loss_func)(state.params)

        new_state = state.apply_gradients(grads=grads)
        if return_value:
            return new_state, val
        return new_state

    if use_parallel:
        return parallelize(train_step)
    else:
        return train_step


class PipelineBasicTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto",
                 namespace=get_ray_namespace_str(
                     prefix=global_config.unittest_ray_namespace_prefix))

    def tearDown(self):
        ray.shutdown()
        time.sleep(1)

    def run_mlp(self,
                manual_pipeline_layer=True,
                test_remat=False,
                pipeline_stage_mode="uniform_layer_gpipe",
                do_numerical_test=True,
                return_value=False):
        virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode=pipeline_stage_mode)

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim

        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=manual_pipeline_layer)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim))
        y = jax.random.normal(rngkey, (batch_size, output_dim))
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        global_config.num_micro_batches = 4
        serial_train_step = get_mlp_train_step(False,
                                               None,
                                               None,
                                               return_value=return_value)
        parallel_train_step = get_mlp_train_step(True,
                                                 manual_pipeline_layer,
                                                 test_remat,
                                                 return_value=return_value)
        executable = parallel_train_step.get_executable(state, batch)

        # Run correctnesss test
        if do_numerical_test:
            expected_new_state = None
            actual_new_state = None
            for i in range(3):
                if i > 0:
                    state = expected_new_state
                if return_value:
                    expected_new_state, expected_val = serial_train_step(
                        state, batch)
                else:
                    expected_new_state, expected_val = serial_train_step(
                        state, batch), 0

                if i > 0:
                    state = actual_new_state
                if return_value:
                    actual_new_state, actual_val = parallel_train_step(
                        state, batch)
                else:
                    actual_new_state, actual_val = parallel_train_step(
                        state, batch), 0

                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1e-3)
                if return_value:
                    assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        executable.shutdown()
        return hlo_text

    def run_n_layer_bert(self,
                         n_layers,
                         manual_pipeline_layer=True,
                         test_remat=False,
                         pipeline_stage_mode="uniform_layer_gpipe",
                         cache_compute_cost=None,
                         forward_stage_layer_ids=None,
                         batch_size=16,
                         seq_len=256,
                         hidden_size=512,
                         num_heads=512 // 64,
                         submesh_shapes=None,
                         do_numerical_test=True,
                         overwrite_global_config_dict=None,
                         virtual_mesh=None,
                         return_value=False):
        num_micro_batch = 2
        if virtual_mesh is None:
            virtual_mesh = DeviceCluster().get_virtual_physical_mesh()
        set_parallelize_options(devices=virtual_mesh,
                                strategy="pipeshard_parallel",
                                pipeline_stage_mode=pipeline_stage_mode,
                                cache_compute_cost=cache_compute_cost,
                                forward_stage_layer_ids=forward_stage_layer_ids,
                                sub_physical_mesh_shapes=submesh_shapes)

        if overwrite_global_config_dict:
            global_config.update_with_dict(overwrite_global_config_dict)

        # Init model and optimizer
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size),
                              dtype=jnp.float32)
        attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)
        model = BertLayerModel(config=BertConfig(hidden_size=hidden_size,
                                                 intermediate_size=hidden_size *
                                                 4,
                                                 num_attention_heads=num_heads,
                                                 num_hidden_layers=n_layers),
                               manual_pipeline_layer=manual_pipeline_layer)
        batch = {"x": x, "y": y, "attention_mask": attention_mask}
        state = create_train_state(rngkey, model, [x, attention_mask])

        # Compile
        global_config.num_micro_batches = num_micro_batch
        serial_train_step = get_bert_layer_train_step(False,
                                                      None,
                                                      None,
                                                      n_layers,
                                                      return_value=return_value)
        parallel_train_step = get_bert_layer_train_step(
            True,
            manual_pipeline_layer,
            test_remat,
            n_layers,
            return_value=return_value)
        executable = parallel_train_step.get_executable(state, batch)

        # Run correctnesss test
        if do_numerical_test:
            expected_new_state = None
            actual_new_state = None
            for i in range(1):
                if i > 0:
                    state = expected_new_state
                if return_value:
                    expected_new_state, expected_val = serial_train_step(
                        state, batch)
                else:
                    expected_new_state, expected_val = serial_train_step(
                        state, batch), 0
                if i > 0:
                    state = actual_new_state
                if return_value:
                    actual_new_state, actual_val = parallel_train_step(
                        state, batch)
                else:
                    actual_new_state, actual_val = parallel_train_step(
                        state, batch), 0
                if return_value:
                    assert_allclose(expected_val, actual_val, 1e-3, 1e-3)
                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1.5e-3)

        hlo_text = executable.get_hlo_text()
        executable.shutdown()
        return hlo_text


def data_loader_test_input_iter_func(start, end, batch_size):
    num_batches = (end - start) // batch_size
    for i in range(num_batches):
        yield (i * np.ones((batch_size, 32), dtype=np.float32), i * np.ones(
            (batch_size,), dtype=np.int32))
