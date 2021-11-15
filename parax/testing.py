"""Utility for testing."""
from collections.abc import Iterable

from flax import linen as nn
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax
from jax.experimental.maps import FrozenDict as FrozenDictJax
import jax.numpy as jnp
import numpy as np
import optax

from parax.model.bert_model import BertConfig, FlaxBertLayer
from parax.model.model_util import TrainState
from parax.pipeline_parallel.automatic_layer_slicing import automatic_layer_slicing
from parax.pipeline_parallel.manual_layer_slicing import manual_layer_slicing, remat
from parax.pipeline_parallel.primitive_def import mark_pipeline

# Store last compiled executables for unit tests.
last_compiled_executable = None
last_compiled_auto_sharding_objective = -1


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


class MLPModel(nn.Module):
    hidden_dim: int
    output_dim: int
    manual_pipeline_layer: bool = True

    @nn.compact
    def __call__(self, x):
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='end')
            mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.Dense(features=self.output_dim, use_bias=True)(x)
        return x


class TwoLayerBertLayerModel(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    manual_pipeline_layer: bool = True

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='start')
        x = self.layer0(x, attention_mask)[0]
        if self.manual_pipeline_layer:
            mark_pipeline(name='1', mark_type='end')
            mark_pipeline(name='2', mark_type='start')
        x = self.layer1(x, attention_mask)[0]
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


def create_train_state(rngkey, model, params):
    params = model.init(rngkey, *params)
    tx = optax.adam(learning_rate=1e-2)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              dynamic_scale=None)
    return state


def decorate_loss_fn(fn, manual_pipeline, use_remat, layer_num):
    if manual_pipeline:
        if use_remat:
            fn = remat(fn)
        return manual_layer_slicing(fn)
    return automatic_layer_slicing(fn,
                                   layer_num=layer_num,
                                   use_pipeline=True,
                                   use_remat=use_remat)
