"""Utilities for testing."""
from functools import partial
import unittest
from collections.abc import Iterable
from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from jax.experimental.maps import FrozenDict as FrozenDictJax
import numpy as np
import optax
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax

from alpa.api import init, shutdown, parallelize, value_and_grad
from alpa.model.bert_model import BertConfig, FlaxBertLayer
from alpa.model.model_util import FlaxBaseModelOutput, DynamicScale, TrainState
from alpa.parallel_method import PipeshardParallel
from alpa.pipeline_parallel.layer_construction import (AutoLayerOption,
                                                       ManualLayerOption)
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.pipeline_parallel.stage_construction import (UniformStageOption,
                                                       StageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption


def assert_allclose(x, y, rtol=1e-4, atol=1e-4):
    """Assert the arrays in x and y are all close."""
    if isinstance(x, (dict, FrozenDictJax, FrozenDictFlax)):
        assert isinstance(y, (dict, FrozenDictJax, FrozenDictFlax))
        assert set(x.keys()) == set(y.keys())
        for k in x.keys():
            assert_allclose(x[k], y[k], rtol, atol)
    elif isinstance(x, Iterable) and not hasattr(x, "__array__"):
        assert isinstance(y, Iterable) and not hasattr(y, "__array__")
        assert len(x) == len(y)
        for x_elt, y_elt in zip(x, y):
            assert_allclose(x_elt, y_elt, rtol, atol)
    elif hasattr(x, "__array__") or np.isscalar(x):
        assert hasattr(y, "__array__") or np.isscalar(y), f"{y}"
        x = np.asarray(x)
        y = np.asarray(y)
        np.testing.assert_allclose(x, y, rtol, atol)
    elif isinstance(x, TrainState):
        assert isinstance(y, TrainState)
        assert_allclose(tree_leaves(x), tree_leaves(y), rtol, atol)
    elif x == y:
        return
    else:
        raise TypeError((type(x), type(y)))


class MLPModel(nn.Module):
    """An MLP model for testing."""
    num_layers: int
    hidden_size: int
    use_bias: bool = True
    add_manual_pipeline_marker: bool = True

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(self.hidden_size, use_bias=self.use_bias)(x)

            if (self.add_manual_pipeline_marker and
                    i == self.num_layers // 2 - 1):
                mark_pipeline_boundary()
        return x


def get_mlp_train_state_and_step(batch_size,
                                 hidden_size,
                                 num_layers=4,
                                 use_bias=True,
                                 add_manual_pipeline_marker=False):
    # Init input batch
    rngkey = jax.random.PRNGKey(0)
    x = jax.random.normal(rngkey, (batch_size, hidden_size))
    y = jax.random.normal(rngkey, (batch_size, hidden_size))
    batch = {"x": x, "y": y}

    # Init model and optimizer
    model = MLPModel(num_layers=num_layers,
                     hidden_size=hidden_size,
                     use_bias=use_bias,
                     add_manual_pipeline_marker=add_manual_pipeline_marker)
    params = model.init(rngkey, batch["x"])
    tx = optax.sgd(learning_rate=1e-2, momentum=0.9)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)

    # Define train step
    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            return jnp.mean((out - batch["y"])**2)

        val, grads = value_and_grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, val

    return state, batch, train_step


class BertLayerModel(nn.Module):
    """A BERT model for testing."""
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    add_manual_pipeline_marker: bool = True

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

            if self.add_manual_pipeline_marker and i != len(self.layers) - 1:
                mark_pipeline_boundary()
        return x


def get_bert_layer_train_state_and_step(batch_size, seq_len, num_layers,
                                        hidden_size, num_heads,
                                        clip_by_global_norm, use_dynamic_scale,
                                        add_manual_pipeline_marker):
    rngkey = jax.random.PRNGKey(0)
    x = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size))
    y = jax.random.normal(rngkey, (batch_size, seq_len, hidden_size))
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int8)
    batch = {"x": x, "y": y, "attention_mask": attention_mask}

    model = BertLayerModel(
        config=BertConfig(hidden_size=hidden_size,
                          intermediate_size=hidden_size * 4,
                          num_attention_heads=num_heads,
                          num_hidden_layers=num_layers),
        add_manual_pipeline_marker=add_manual_pipeline_marker)
    params = model.init(rngkey, batch["x"], batch["attention_mask"])

    if clip_by_global_norm:
        tx = optax.chain(optax.clip_by_global_norm(0.05),
                         optax.adam(learning_rate=1e-2))
    else:
        tx = optax.adam(learning_rate=1e-2)

    if use_dynamic_scale:
        use_master_copy = False
        dynamic_scale = DynamicScale()
    else:
        dynamic_scale = None
        use_master_copy = False

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=dynamic_scale,
                              use_master_copy=use_master_copy)

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"], batch["attention_mask"])
            loss = jnp.mean((out - batch["y"])**2)
            return loss

        dynamic_scale = state.dynamic_scale
        if dynamic_scale:
            grad_fn = dynamic_scale.value_and_grad(loss_func)
            dynamic_scale, is_fin, val, grads = grad_fn(state.params)
        else:
            grad_fn = value_and_grad(loss_func)
            val, grads = grad_fn(state.params)

        new_state = state.apply_gradients(grads=grads)

        if dynamic_scale:
            new_state = new_state.replace(
                opt_state=jax.tree_map(partial(jnp.where, is_fin),
                                       new_state.opt_state, state.opt_state),
                params=jax.tree_map(partial(jnp.where, is_fin),
                                    new_state.params, state.params),
                master_copy=jax.tree_map(partial(jnp.where,
                                                 is_fin), new_state.master_copy,
                                         state.master_copy),
                dynamic_scale=dynamic_scale)
        return new_state, val

    return state, batch, train_step


def create_train_state(rngkey, model, inputs):
    params = model.init(rngkey, *inputs)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


def mlp_inference_step(state, batch):
    out = state.apply_fn(state.params, batch["x"])
    loss = jnp.mean((out - batch["y"])**2)
    return out, loss


def bert_layer_collection_inference_step(state, batch):
    out = state.apply_fn(state.params,
                         batch["x"],
                         batch["attention_mask"],
                         output_attentions=True,
                         output_hidden_states=True)
    loss = jnp.mean((out.last_hidden_state - batch["y"])**2)
    # FIXME(yonghao): Otherwise, the first hidden state is an input,
    #   but we do not support outputing an input(not batch-related
    #   outputs).
    out = FlaxBaseModelOutput(last_hidden_state=out.last_hidden_state,
                              hidden_states=out.hidden_states[1:],
                              attentions=out.attentions)
    return out, loss


class PipelineBasicTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def tearDown(self):
        shutdown()

    def run_mlp(self,
                manual_pipeline_layer: bool = True,
                use_remat: bool = False,
                stage_option: Optional[StageOption] = None,
                as_option: Optional[AutoShardingOption] = None,
                do_numerical_test: bool = True):
        method = PipeshardParallel(
            num_micro_batches=4,
            default_auto_sharding_option=as_option or AutoShardingOption(),
            layer_option=ManualLayerOption(remat_layer=use_remat)
            if manual_pipeline_layer else AutoLayerOption(
                layer_num=2,
                remat_mode="coarse_grained_remat" if use_remat else "none"),
            stage_option=stage_option or UniformStageOption())

        # Init model
        state, batch, train_step = get_mlp_train_state_and_step(
            batch_size=64,
            hidden_size=16,
            num_layers=4,
            add_manual_pipeline_marker=manual_pipeline_layer)

        # Compile
        serial_train_step = train_step
        parallel_train_step = parallelize(train_step, method=method)
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
                actual_new_state, actual_val = parallel_train_step(state, batch)

                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1e-3)
                assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text

    def run_n_layer_bert(self,
                         num_layers,
                         batch_size=16,
                         seq_len=256,
                         hidden_size=512,
                         num_heads=512 // 64,
                         use_remat=False,
                         clip_by_global_norm=False,
                         use_dynamic_scale=False,
                         inject_train_step=None,
                         manual_pipeline_layer=True,
                         stage_option: Optional[StageOption] = None,
                         as_option: Optional[AutoShardingOption] = None,
                         do_numerical_test: bool = True):
        method = PipeshardParallel(
            num_micro_batches=4,
            default_auto_sharding_option=as_option or AutoShardingOption(),
            layer_option=ManualLayerOption(remat_layer=use_remat)
            if manual_pipeline_layer else AutoLayerOption(
                layer_num=num_layers,
                remat_mode="coarse_grained_remat" if use_remat else "none"),
            stage_option=stage_option or UniformStageOption())

        # Init model
        state, batch, train_step = get_bert_layer_train_state_and_step(
            batch_size=batch_size,
            seq_len=seq_len,
            num_layers=num_layers,
            hidden_size=hidden_size,
            num_heads=num_heads,
            clip_by_global_norm=clip_by_global_norm,
            use_dynamic_scale=use_dynamic_scale,
            add_manual_pipeline_marker=manual_pipeline_layer)
        if inject_train_step is not None:
            assert isinstance(inject_train_step, Callable)
            train_step = inject_train_step

        # Compile
        serial_train_step = train_step
        parallel_train_step = parallelize(train_step, method=method)
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

                actual_new_state, actual_val = parallel_train_step(state, batch)

                assert_allclose(expected_new_state.params,
                                actual_new_state.params, 1e-3, 1.5e-3)
                assert_allclose(expected_val, actual_val, 1e-3, 1e-3)

        hlo_text = executable.get_hlo_text()
        return hlo_text


def data_loader_input_iter_func(start, end, batch_size):
    """A data loader function for testing."""
    dataset_x = np.arange(1024 * 32).reshape(-1, 32).astype(np.float32)
    dataset_y = np.arange(1024).astype(np.int32)

    num_batches = (end - start) // batch_size

    for i in range(num_batches):
        idx = start + i * batch_size
        yield dataset_x[idx:idx + batch_size], dataset_y[idx:idx + batch_size]


class HloParser:

    @staticmethod
    def get_param_line(text: str):
        text = text[text.find("ENTRY"):]
        text = text[:text.find("\n")]
        return text

    @staticmethod
    def get_root_line(text: str):
        text = text[text.find("ENTRY"):]
        text = text[text.find("ROOT"):]
        text = text[:text.find("\n")]
        return text

    @staticmethod
    def parse_param_shapes(text: str):
        # the first one is "ENTRY %xxx ("
        params = text.split("param")[1:]
        shapes = tuple(map(lambda x: x[x.find("f32"):x.find("]") + 1], params))
        return shapes

    @staticmethod
    def parse_root_shapes(text: str):
        tuple_shape = text[text.find("=") + 2:text.find("tuple(")]
        # the last one is ')'
        shapes = tuple_shape.split("0}")[:-1]
        shapes = tuple(map(lambda x: x[x.find("f32"):x.find("{")], shapes))
        return shapes
