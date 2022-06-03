"""Utilities for testing."""
import unittest
from collections.abc import Iterable
from typing import Optional

import jax
import jax.numpy as jnp
from jax.experimental.maps import FrozenDict as FrozenDictJax
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax

from alpa.api import init, shutdown, parallelize, grad, value_and_grad
from alpa.model.bert_model import BertConfig, FlaxBertLayer
from alpa.model.model_util import FlaxBaseModelOutput, TrainState
from alpa.parallel_method import PipeshardParallel
from alpa.pipeline_parallel.layer_construction import (
    automatic_layer_construction, manual_layer_construction)
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.pipeline_parallel.stage_construction import UniformStageOption, StageOption
from alpa.shard_parallel.auto_sharding import AutoShardingOption


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
        assert hasattr(y, '__array__') or np.isscalar(y), f"{y}"
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
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        if self.manual_pipeline_layer:
            mark_pipeline_boundary()
        x = nn.Dense(features=self.hidden_dim, use_bias=self.use_bias)(x)
        x = nn.Dense(features=self.output_dim, use_bias=self.use_bias)(x)
        return x


class BertLayerModel(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    manual_pipeline_layer: bool = True

    def setup(self):
        # pylint: disable=attribute-defined-outside-init
        self.layers = [
            FlaxBertLayer(config=self.config, dtype=self.dtype)
            for _ in range(self.config.num_hidden_layers)
        ]

    def __call__(self, x, attention_mask):
        for i, layer in enumerate(self.layers):
            layer_outputs = layer(x, attention_mask)
            x = layer_outputs[0]
            if self.manual_pipeline_layer and i != len(self.layers) - 1:
                mark_pipeline_boundary()
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
        return manual_layer_construction(fn,
                                         remat_layer=use_remat)
    return automatic_layer_construction(fn,
                                        remat_layer=use_remat,
                                        layer_num=layer_num)


def get_mlp_train_step(parallel_method,
                       manual_pipeline_layer,
                       use_remat,
                       use_value_and_grad):

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        if parallel_method:
            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         use_remat, 2)
            if use_value_and_grad:
                val, grads = value_and_grad(loss_func)(state.params)
            else:
                grads = grad(loss_func)(state.params)
                val = jax.tree_leaves(grads)[0]
        else:
            if use_value_and_grad:
                val, grads = jax.value_and_grad(loss_func)(state.params)
            else:
                grads = jax.grad(loss_func)(state.params)
                val = jax.tree_leaves(grads)[0]

        new_state = state.apply_gradients(grads=grads)
        return new_state, val

    if parallel_method:
        return parallelize(train_step, method=parallel_method)
    else:
        return train_step


def get_mlp_inference_step(parallel_method,
                           manual_pipeline_layer):

    def inference_step(state, batch):

        def forward(params):
            out = state.apply_fn(params, batch["x"])
            loss = jnp.mean((out - batch["y"])**2)
            return out, loss

        if parallel_method:
            forward = decorate_loss_fn(forward, manual_pipeline_layer,
                                       False, 2)

        out = forward(state.params)
        return out

    if parallel_method:
        return parallelize(inference_step, donate_argnums=(),
                           method=parallel_method)
    else:
        return inference_step


def get_bert_layer_collection_inference_step(parallel_method,
                                             manual_pipeline_layer,
                                             n_layers):

    def inference_step(state, batch):

        def forward(params):
            out = state.apply_fn(params,
                                 batch["x"],
                                 batch["attention_mask"],
                                 output_attentions=True,
                                 output_hidden_states=True)
            loss = jnp.mean((out.last_hidden_state - batch["y"])**2)
            # FIXME(yonghao): Otherwise, the first hidden state is an input,
            # but we do not support outputing an input(not batch-related outputs).
            out = FlaxBaseModelOutput(last_hidden_state=out.last_hidden_state,
                                      hidden_states=out.hidden_states[1:],
                                      attentions=out.attentions)
            return out, loss

        if parallel_method:
            forward = decorate_loss_fn(forward, manual_pipeline_layer,
                                       False, 2)
        out = forward(state.params)
        return out

    if parallel_method:
        return parallelize(inference_step, donate_argnums=(),
                           method=parallel_method)
    else:
        return inference_step


def get_bert_layer_train_step(parallel_method,
                              manual_pipeline_layer,
                              use_remat,
                              num_layers,
                              use_value_and_grad,
                              decorate_loss=None):
    if decorate_loss is None:
        decorate_loss = parallel_method is not None

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"], batch["attention_mask"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        if decorate_loss:
            loss_func = decorate_loss_fn(loss_func, manual_pipeline_layer,
                                         use_remat, num_layers)
            if use_value_and_grad:
                val, grads = value_and_grad(loss_func)(state.params)
            else:
                grads = grad(loss_func)(state.params)
                val = jax.tree_leaves(grads)[0]
        else:
            if use_value_and_grad:
                val, grads = jax.value_and_grad(loss_func)(state.params)
            else:
                grads = jax.grad(loss_func)(state.params)
                val = jax.tree_leaves(grads)[0]

        new_state = state.apply_gradients(grads=grads)
        return new_state, val

    if parallel_method:
        return parallelize(train_step, method=parallel_method)
    else:
        return train_step


class PipelineBasicTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def run_mlp(self,
                manual_pipeline_layer: bool = True,
                use_remat: bool = False,
                use_value_and_grad: bool = False,
                stage_option: Optional[StageOption] = None,
                as_option: Optional[AutoShardingOption] = None,
                do_numerical_test: bool = True):
        method = PipeshardParallel(num_micro_batches=4)
        method.stage_option = stage_option or UniformStageOption()
        method.as_option = as_option or AutoShardingOption()

        # Init model and optimizer
        batch_size = 64
        hidden_dim = 16
        input_dim = output_dim = hidden_dim

        model = MLPModel(hidden_dim=hidden_dim,
                         output_dim=output_dim,
                         manual_pipeline_layer=manual_pipeline_layer)
        rngkey = jax.random.PRNGKey(0)
        x = jax.random.normal(rngkey, (batch_size, input_dim), jnp.float32)
        y = jax.random.normal(rngkey, (batch_size, output_dim), jnp.float32)
        batch = {'x': x, 'y': y}
        state = create_train_state(rngkey, model, [x])

        # Compile
        serial_train_step = get_mlp_train_step(None,
                                               None,
                                               None,
                                               use_value_and_grad)
        parallel_train_step = get_mlp_train_step(method,
                                                 manual_pipeline_layer,
                                                 use_remat,
                                                 use_value_and_grad)
        executable = parallel_train_step.get_executable(state, batch)

        # Run correctnesss test
        if do_numerical_test:
            expected_new_state = None
            actual_new_state = None
            for i in range(3):
                if i > 0:
                    state = expected_new_state
                expected_new_state, expected_val = serial_train_step(
                    state, batch)

                if i > 0:
                    state = actual_new_state
                actual_new_state, actual_val = parallel_train_step(
                    state, batch)

                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1e-3)
                assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text

    def run_n_layer_bert(self,
                         n_layers,
                         batch_size=16,
                         seq_len=256,
                         hidden_size=512,
                         num_heads=512 // 64,
                         use_remat=False,
                         use_value_and_grad=False,
                         manual_pipeline_layer=True,
                         stage_option: Optional[StageOption] = None,
                         as_option: Optional[AutoShardingOption] = None,
                         do_numerical_test: bool = True):
        method = PipeshardParallel(num_micro_batches=2)
        method.stage_option = stage_option or UniformStageOption()
        method.as_option = as_option or AutoShardingOption()

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
        serial_train_step = get_bert_layer_train_step(None,
                                                      None,
                                                      None,
                                                      n_layers,
                                                      use_value_and_grad)
        parallel_train_step = get_bert_layer_train_step(method,
                                                        manual_pipeline_layer,
                                                        use_remat, n_layers,
                                                        use_value_and_grad)
        executable = parallel_train_step.get_executable(state, batch)

        # Run correctnesss test
        if do_numerical_test:
            expected_new_state = None
            actual_new_state = None
            for i in range(1):
                if i > 0:
                    state = expected_new_state
                expected_new_state, expected_val = serial_train_step(
                    state, batch)

                if i > 0:
                    state = actual_new_state

                actual_new_state, actual_val = parallel_train_step(
                    state, batch)

                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1.5e-3)
                assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text


def data_loader_test_input_iter_func(start, end, batch_size):
    num_batches = (end - start) // batch_size
    for i in range(num_batches):
        yield (i * np.ones((batch_size, 32), dtype=np.float32), i * np.ones(
            (batch_size,), dtype=np.int32))
