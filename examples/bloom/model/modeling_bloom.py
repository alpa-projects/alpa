# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and Bigscience Workshop. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Flax BLOOM model."""

import math
from functools import partial
from typing import Callable, Optional, Tuple, Dict, Sequence

import dataclasses
from dataclasses import dataclass
import itertools
import os

import numpy as np

import jax
from jax import lax
from jax.interpreters import pxla
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
import jaxlib.xla_extension as jax_xla

import flax
import flax.linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.linen.partitioning import scan_with_axes
from flax.traverse_util import flatten_dict, unflatten_dict

from transformers.modeling_flax_utils import FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging

import alpa
from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              MeshHostWorker, create_remote_array_refs)
from alpa.model.model_util import ModelOutput
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary

from tqdm import tqdm


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/bloom"
_CONFIG_FOR_DOC = "BloomConfig"
_TOKENIZER_FOR_DOC = "BloomTokenizerFast"

@dataclass(frozen=True)
class BloomConfig:
    model_type: str = "bloom"
    vocab_size: int = 250880
    max_target_positions: int = 64
    hidden_size: int = 64
    n_head: int = 8
    num_hidden_layers: int = 2
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = False
    bos_token_id: int = 1
    eos_token_id: int = 2
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    pretraining_tp: int = 1  # TP rank used when training with megatron
    slow_but_exact: bool = False
    tie_word_embeddings: bool = True
    dtype: any = jnp.float16
    pad: int = 1

@flax.struct.dataclass
class BloomModelOutput(ModelOutput):
    last_hidden_state: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None

@flax.struct.dataclass
class BloomLMOutput(ModelOutput):
    logits: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None

BLOOM_START_DOCSTRING = r"""
    This model inherits from [`FlaxPreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a Flax Linen
    [flax.nn.Module](https://flax.readthedocs.io/en/latest/_autosummary/flax.nn.module.html) subclass. Use it as a
    regular Flax Module and refer to the Flax documentation for all matter related to general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~FlaxPreTrainedModel.from_pretrained`] method to load the model weights.
        dtype (`jax.numpy.dtype`, *optional*, defaults to `jax.numpy.float16`):
            The data type of the computation. Can be one of `jax.numpy.float16`, `jax.numpy.float16` (on GPUs) and
            `jax.numpy.bfloat16` (on TPUs).
            This can be used to enable mixed-precision training or half-precision inference on GPUs or TPUs. If
            specified all the computation will be performed with the given `dtype`.
            **Note that this only specifies the dtype of the computation and does not influence the dtype of model
            parameters.**
            If you wish to change the dtype of the model parameters, see [`~FlaxPreTrainedModel.to_fp16`] and
            [`~FlaxPreTrainedModel.to_bf16`].
"""

BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`numpy.ndarray` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length`. Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`BloomTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`numpy.ndarray` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        past_key_values (`Dict[str, np.ndarray]`, *optional*, returned by `init_cache` or when passing previous `past_key_values`):
            Dictionary of pre-computed hidden-states (key and values in the attention blocks) that can be used for fast
            auto-regressive decoding. Pre-computed key and value hidden-states are of shape *[batch_size, max_length]*.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

def build_alibi_tensor_flax(attention_mask, n_head, dtype):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # Note: alibi will be added to the attention bias that is applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=n_head, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcast correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_flax_t5.py#L426
    # batch_size = 1, n_head = n_head, query_length
    # shape of attention_mask: [B, 1, 1, S_max]
    batch_size = attention_mask.shape[0]
    key_length = attention_mask.shape[-1]
    attention_mask = attention_mask.reshape((batch_size, key_length))
    num_heads = n_head
    query_length = 1

    slopes = jnp.array(get_slopes(n_head))[None, :, None, None].astype(dtype)
    arange_tensor = attention_mask.cumsum(-1, dtype=dtype)[:, None, None, :] - 1

    slopes_broadcast = jnp.broadcast_to(slopes, (batch_size, num_heads, query_length, key_length))
    arange_broadcast = jnp.broadcast_to(arange_tensor, (batch_size, num_heads, query_length, key_length))

    alibi = slopes_broadcast * arange_broadcast
    return alibi

class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        dense = partial(
            nn.Dense,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(self.config.initializer_range),
        )

        self.query_key_value = dense(self.hidden_size * 3)
        self.dense = dense(self.hidden_size)
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    def __call__(
        self,
        hidden_states,
        residual,
        alibi,
        layer_past=None,
        attention_mask=None,
        attention_cache=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        layer_number: int = None,
    ):
        # This chunk verified to be working
        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        fused_qkv = self.query_key_value(hidden_states)
        fused_qkv = fused_qkv.reshape(fused_qkv.shape[:-1] + (self.num_heads, self.head_dim * 3))
        query, key, value = jnp.split(fused_qkv, 3, axis=-1)
        key_len = attention_mask.shape[-1]
        causal_attention_mask = make_causal_mask(jnp.ones((batch_size, key_len)), dtype="bool")

        # for fast decoding causal attention mask should be shifted
        if attention_cache:
            causal_attention_mask_shift = attention_cache[2][0]
        else:
            causal_attention_mask_shift = 0
        
        # fast decoding for generate requires special attention_mask
        if attention_cache:
            max_decoder_length = attention_cache[0].shape[1]
            causal_attention_mask = jax.lax.dynamic_slice(
                causal_attention_mask,
                (0, 0, causal_attention_mask_shift, 0),
                (1, 1, seq_length, max_decoder_length),
            )

        attention_mask = combine_masks(attention_mask, causal_attention_mask)

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if attention_cache:
            cache_key, cache_value, cache_index = attention_cache
            *batch_dims, max_length, num_heads, depth_per_head = cache_key.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index[0]
            indices = (0, cur_index, 0, 0)
            key = lax.dynamic_update_slice(cache_key, key, indices)
            value = lax.dynamic_update_slice(cache_value, value, indices)
            cache_key = key
            cache_value = value
            num_updated_cache_vectors = query.shape[1]
            # A line added from opt_model
            attention_cache = key, value, cache_index + num_updated_cache_vectors
            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(max_length) < cur_index + num_updated_cache_vectors,
                tuple(batch_dims) + (1, num_updated_cache_vectors, max_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)

        dropout_rng = None
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # transform boolean mask into float mask
        mask_value = jnp.finfo(self.dtype).min
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, mask_value).astype(self.dtype),
        )

        attention_bias = attention_bias + alibi

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_dropout,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = attn_output.reshape(hidden_states.shape[:2] + (self.hidden_size,))
        attn_output = self.dense(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        attn_output = attn_output + residual

        outputs = (attn_output, attention_cache,
                   attn_weights) if output_attentions else (attn_output,
                                                            attention_cache)
        return outputs


class BloomGELU(nn.Module):
    def setup(self):
        self.dtype = jnp.float32

    def __call__(self, x):
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        hidden_size = self.config.hidden_size

        self.pretraining_tp = self.config.pretraining_tp
        self.slow_but_exact = self.config.slow_but_exact

        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.dense_h_to_4h = nn.Dense(4 * hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.dense_4h_to_h = nn.Dense(hidden_size, dtype=self.dtype, kernel_init=kernel_init)
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        self.act = BloomGELU()

    def __call__(self, hidden_states, residual, deterministic: bool = True):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)

        intermediate_output = self.dense_4h_to_h(hidden_states)

        intermediate_output = intermediate_output + residual
        hidden_states = self.hidden_dropout(intermediate_output, deterministic=deterministic)

        return hidden_states


class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.input_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=jnp.float32)

        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype)
        self.post_attention_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype)

        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        attention_cache=None,
        layer_number: int = None,
        layer_past=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        layernorm_output = self.input_layernorm(hidden_states)
        # layer norm before saving residual if config calls for it
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # self-attention
        attn_outputs = self.self_attention(
            layernorm_output,
            residual=residual,
            alibi=alibi,
            layer_past=layer_past,
            attention_mask=attention_mask,
            attention_cache=attention_cache,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            layer_number=layer_number,
        )
        attention_output = attn_outputs[0]
        attention_cache = attn_outputs[1]

        post_layernorm = self.post_attention_layernorm(attention_output)

        # set residual based on config
        if self.apply_residual_connection_post_layernorm:
            residual = post_layernorm
        else:
            residual = attention_output

        output = self.mlp(post_layernorm, residual, deterministic=deterministic)

        outputs = (output, attention_cache)
        if output_attentions:
            outputs += (attn_outputs[2],)
        return outputs

class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16

    @nn.compact
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        attention_cache=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        new_attention_cache = () if attention_cache is not None else None

        for layer_number in range(self.config.num_hidden_layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_attention_cache = None
            if attention_cache is not None:
                layer_attention_cache = attention_cache[layer_number]
            layer_outputs = FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype)(
                hidden_states,
                alibi=alibi,
                attention_mask=attention_mask,
                attention_cache=layer_attention_cache,
                deterministic=deterministic,
                init_cache=init_cache,
                output_attentions=output_attentions,
                layer_number=layer_number,
            )
            hidden_states = layer_outputs[0]
            if attention_cache is not None:
                new_attention_cache += (layer_outputs[1],)

            if output_attentions:
                all_attentions += (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)
        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return BloomModelOutput(last_hidden_state=hidden_states,
                              hidden_states=all_hidden_states,
                              attentions=all_attentions,
                              attention_cache=new_attention_cache)


class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.embed_dim = self.config.hidden_size

        embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range)

        # word embeddings (no positional embedding layer)
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=embedding_init,
            dtype=self.dtype,
        )

        # post-embedding layernorm
        self.word_embeddings_layernorm = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

        # transformer layers
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype)

        # final layernorm
        self.ln_f = nn.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        attention_cache=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # do post-embedding layernorm
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        batch_size, curr_seq_len, _ = hidden_states.shape

        # build alibi depending on `attention_mask`
        alibi = build_alibi_tensor_flax(attention_mask, self.config.n_head, hidden_states.dtype)

        outputs = self.h(
            hidden_states,
            alibi=alibi,
            attention_mask=attention_mask,
            attention_cache=attention_cache,
            deterministic=deterministic,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states + (hidden_states,)
            outputs = BloomModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=outputs.attentions, attention_cache=outputs.attention_cache)
        else:
            outputs = BloomModelOutput(last_hidden_state=hidden_states, hidden_states=outputs.hidden_states, attentions=outputs.attentions, attention_cache=outputs.attention_cache)

        if not return_dict:
            return (hidden_states,) + outputs[1:]

        return outputs


@add_start_docstrings(
    "The bare Bloom Model transformer outputting raw hidden-states without any specific head on top.",
    BLOOM_START_DOCSTRING,
)


class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype)
        self.lm_head = nn.Dense(
            self.config.vocab_size,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        attention_cache=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            attention_cache=attention_cache,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformer.variables["params"]["word_embeddings"]["embedding"].T
            lm_logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            lm_logits = self.lm_head(hidden_states)

        if not return_dict:
            return (lm_logits,) + outputs[1:]

        return BloomLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions, attention_cache=outputs.attention_cache)


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)

def get_bloom_config(name, **kwargs):
    if name == "350M":
        config = BloomConfig(
            hidden_size = 1024,
            n_head = 16,
            num_hidden_layers = 24,
            pretraining_tp = 1,
            use_cache = True
        )
    elif name == "760M":
        config = BloomConfig(
            hidden_size = 1536,
            n_head = 16,
            num_hidden_layers = 24,
            pretraining_tp = 1,
            use_cache = True
        )
    elif name == "1.3B":
        config = BloomConfig(
            hidden_size = 2048,
            n_head = 16,
            num_hidden_layers = 24,
            pretraining_tp = 2,
            use_cache = True
        )
    elif name == "2.5B":
        config = BloomConfig(
            hidden_size = 2560,
            n_head = 32,
            num_hidden_layers = 30,
            pretraining_tp = 4,
            use_cache = True
        )
    elif name == "6.3B":
        config = BloomConfig(
            hidden_size = 4096,
            n_head = 32,
            num_hidden_layers = 30,
            pretraining_tp = 4,
            use_cache = True
        )
    elif name == "176B":
        config = BloomConfig(
            hidden_size = 14336,
            n_head = 112,
            num_hidden_layers = 70,
            pretraining_tp = 4,
            use_cache = True
        )
    else:
        raise ValueError()

    return dataclasses.replace(config, **kwargs)

def init_model_aval(config):
    """Initialize model with parameters with abstract values (shape-only arrays)."""
    model = FlaxBloomForCausalLMModule(config, dtype=config.dtype)
    rngkey = jax.core.ShapedArray((2,), jnp.uint32)
    input_ids = jax.core.ShapedArray((1, 64), jnp.int32)
    attention_mask = jax.core.ShapedArray((1, 1, 1, 64), jnp.int32)
    params = jax.eval_shape(model.init, rngkey, input_ids, attention_mask=attention_mask)
    params = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, config.dtype),
                          params)
    return model, params


def init_cache_aval(config, batch_size):
    """Initialize cache with abstract values (shape-only arrays)."""
    dtype = config.dtype
    head_dim = config.hidden_size // config.n_head

    all_cache = []
    for _ in range(config.num_hidden_layers):
        layer_cache = (
            jax.core.ShapedArray((batch_size, config.max_target_positions,
                                  config.n_head, head_dim),
                                 dtype),
            jax.core.ShapedArray((batch_size, config.max_target_positions,
                                  config.n_head, head_dim),
                                 dtype),
            jax.core.ShapedArray((batch_size,), jnp.int32),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def init_mask_aval(config, batch_size):
    """Initialize attention mask with abstract values (shape-only arrays)."""
    mask = jax.core.ShapedArray((batch_size, 1, 1, config.max_target_positions), dtype=np.int8)
    return mask


def init_cache_np(config, batch_size):
    """Init cache with numpy arrays."""
    np_dtype = config.dtype
    head_dim = config.hidden_size // config.n_head

    all_cache = []
    for i in range(config.num_hidden_layers):
        layer_cache = (
            np.zeros((batch_size, config.max_target_positions,
                      config.n_head, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size, config.max_target_positions,
                      config.n_head, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size,), np.int32),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def inference_step_no_cache(params, batch, apply_func):
    # logits = apply_func(params, batch["input_ids"], batch["position_ids"])[0]
    logits = apply_func(params, batch["input_ids"])[0]
    return logits

def load_params_np(params, path, config, dummy=False):
    """Load parameters with numpy arrays."""
    if dummy:
        np_dtype = config.dtype
        return jax.tree_map(lambda x: np.full(x.shape, 1e-9, np_dtype), params)

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array, is_position_embedding=False):
        param_dict = params
        param_keys = param_key.split('.')
        for i, key in enumerate(param_keys):
            if i == len(param_keys) - 1:
                if dummy:
                    param_dict[key] = jax.core.ShapedArray(
                        param_dict[key].shape, param_dict[key].dtype)
                else:
                    if not is_position_embedding:
                        assert param_dict[key].shape == loaded_array.shape, (
                                f"{param_dict[key].shape} vs. {loaded_array.shape}")
                    else:
                        shape = param_dict[key].shape
                        if shape != loaded_array.shape:
                            assert shape[1] == loaded_array.shape[1]
                            loaded_array = loaded_array[:shape[0], :]
                    param_dict[key] = loaded_array
            else:
                param_dict = param_dict[key]

    params = params.unfreeze()
    load_param("params.transformer.ln_f.scale",
               load_array("ln_f.weight"))
    load_param("params.transformer.ln_f.bias",
               load_array("ln_f.bias"))
    load_param("params.transformer.word_embeddings.embedding",
               load_array("word_embeddings.weight"))
    load_param("params.transformer.word_embeddings_layernorm.scale",
                load_array("word_embeddings_layernorm.weight"))
    load_param("params.transformer.word_embeddings_layernorm.bias",
                load_array("word_embeddings_layernorm.bias"))
    for i in tqdm(range(config.num_hidden_layers)):
        param_prefix = f"params.transformer.h.{i}."
        load_prefix = f"h.{i}."
        # Attention weights
        load_param(param_prefix + "self_attention.query_key_value.kernel",
                   load_array(load_prefix + "self_attention.query_key_value.weight").transpose())
        load_param(param_prefix + "self_attention.query_key_value.bias",
                   load_array(load_prefix + "self_attention.query_key_value.bias").transpose())
        load_param(param_prefix + "input_layernorm.scale",
                   load_array(load_prefix + "input_layernorm.weight"))
        load_param(param_prefix + "input_layernorm.bias",
                   load_array(load_prefix + "input_layernorm.bias"))
        load_param(param_prefix + "self_attention.dense.kernel",
                   load_array(load_prefix + "self_attention.dense.weight"))
        load_param(param_prefix + "self_attention.dense.bias",
                   load_array(load_prefix + "self_attention.dense.bias"))
        load_param(param_prefix + "post_attention_layernorm.scale",
                   load_array(load_prefix + "post_attention_layernorm.weight"))
        load_param(param_prefix + "post_attention_layernorm.bias",
                   load_array(load_prefix + "post_attention_layernorm.bias"))
        # MLP weights
        load_param(param_prefix + "mlp.dense_h_to_4h.kernel",
                   np.transpose(load_array(load_prefix + "mlp.dense_h_to_4h.weight")))
        load_param(param_prefix + "mlp.dense_h_to_4h.bias",
                   np.transpose(load_array(load_prefix + "mlp.dense_h_to_4h.bias")))
        load_param(param_prefix + "mlp.dense_4h_to_h.kernel",
                   np.transpose(load_array(load_prefix + "mlp.dense_4h_to_h.weight")))
        load_param(param_prefix + "mlp.dense_4h_to_h.bias",
                   np.transpose(load_array(load_prefix + "mlp.dense_4h_to_h.bias")))

    return flax.core.freeze(params)

def get_jax_executable(config: BloomConfig,
                       encoder_seq_lengths: Sequence[int],
                       output_attentions: bool = False,
                       output_hidden_states:bool = False):
    """Get a single-gpu executable."""
    model, params = init_model_aval(config)

    @jax.jit
    def inference_step(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             attention_cache=batch["cache"],
                             attention_mask=batch["mask"],
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states)
        return output

    executables = {}
    for length in encoder_seq_lengths:
        executables[length] = inference_step
    return executables, params

def get_pipeshard_executable(config: BloomConfig,
                             batch_size: int,
                             encoder_seq_lengths: Sequence[int],
                             num_micro_batches: int = 1,
                             output_attentions: bool = False,
                             output_hidden_states: bool = False,
                             autoregressive: bool = True):
    """Get a parallel executable."""
    # Init model
    model, params = init_model_aval(config)

    # Parallelize
    method = alpa.PipeshardParallel(
        num_micro_batches=num_micro_batches,
        pipeline_schedule="inference",
        layer_option="manual",
        default_auto_sharding_option=alpa.AutoShardingOption(
            # Force operator model parallel
            force_batch_dim_to_mesh_dim=None if batch_size == 1 else 0,
            # Disabling all-to-all and all-gather generates better intra-op strategies.
            allow_all_to_all=False,
            allow_all_gather=False,
        ))

    if autoregressive:

        def inference_step_with_cache(params, batch):
            output = model.apply(
                params,
                batch["input_ids"],
                attention_cache=batch["cache"],
                attention_mask=batch["mask"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)
            return output

        alpa.global_config.always_donate_micro_batch_vars = False

        cache = init_cache_aval(config, batch_size)
        mask = init_mask_aval(config, batch_size)

        executables = {}

        # Compile an executable with sequence length 1
        executable = alpa.parallelize(
            inference_step_with_cache, batch_argnums=(1,),
            method=method).get_executable(
                params, {
                    "input_ids":
                        jax.core.ShapedArray((batch_size, 1), jnp.int32),
                    # "position_ids":
                    #     jax.core.ShapedArray((batch_size, 1), jnp.int32),
                    "cache":
                        cache,
                    "mask":
                        mask,
                })
        executable.dump_debug_info("tmp_executable_1")
        executables[1] = executable

        # Create another parallel method with assigned input sharding specs
        method_with_input_sharding = alpa.PipeshardParallel(
            num_micro_batches=num_micro_batches,
            pipeline_schedule="inference",
            layer_option="manual",
            default_auto_sharding_option=alpa.AutoShardingOption(
                enable_auto_sharding=False,
            ),
            stage_input_shardings=executable.stage_input_shard_specs)

        # Compile other executables
        for seq_len in encoder_seq_lengths:
            executable = alpa.parallelize(
                inference_step_with_cache,
                batch_argnums=(1,),
                method=method_with_input_sharding).get_executable(
                    params, {
                        "input_ids":
                            jax.core.ShapedArray(
                                (batch_size, seq_len), jnp.int32),
                        # "position_ids":
                        #     jax.core.ShapedArray(
                        #         (batch_size, seq_len), jnp.int32),
                        "cache":
                            cache,
                        "mask":
                            mask,
                    })
            executable.dump_debug_info("tmp_executable_%d" % seq_len)
            executables[seq_len] = executable
        return executables, params
    else:
        assert len(encoder_seq_lengths) == 1
        seq_len = encoder_seq_lengths[0]

        @alpa.parallelize(batch_argnums=(1,), method=method)
        def inference_step(params, batch):
            output = model.apply(
                params,
                batch["input_ids"],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states)
            return output

        assert batch_size % num_micro_batches == 0, "cannot divide batch_size by num_micro_batches"
        micro_batch_size = batch_size // num_micro_batches

        executable = inference_step.get_executable(
            params, {
                "input_ids":
                    jax.core.ShapedArray(
                        (batch_size, seq_len), jnp.int32),
                # "position_ids":
                #     jax.core.ShapedArray(
                #         (batch_size, seq_len), jnp.int32),
            })

        executable.dump_debug_info("tmp")
    return {seq_len: executable}, params


def load_bloom_params_worker_func(self, path, prefix_to_idx, config, shapes,
                                uuids, indices, mesh_ids):
    """The worker function to load Bloom parameters."""

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array, is_position_embedding=False):
        i = prefix_to_idx[param_key]

        for j in range(len(mesh_ids[i])):
            if self.mesh_id != mesh_ids[i][j]:
                continue

            if not is_position_embedding:
                assert shapes[i][j] == loaded_array.shape
            else:
                if shapes[i][j] != loaded_array.shape:
                    assert shapes[i][j][1] == loaded_array.shape[1]
                    loaded_array = loaded_array[:shapes[i][j][0], :]
            uuid = uuids[i][j]
            datas = []
            for k in range(len(self.local_devices)):
                idx = self.host_id * len(self.local_devices) + k
                datas.append(loaded_array[indices[i][j][idx]])
            self.put_buffers(uuid, datas)
    layers_per_stage = config.decoder_layers // config.num_pp_stages

    load_param("params.transformer.ln_f.scale",
               load_array("ln_f.weight"))
    load_param("params.transformer.ln_f.bias",
               load_array("ln_f.bias"))
    load_param("params.transformer.word_embeddings.embedding",
               load_array("word_embeddings.weight"))
    load_param("params.transformer.word_embeddings_layernorm.scale",
                load_array("word_embeddings_layernorm.weight"))
    load_param("params.transformer.word_embeddings_layernorm.bias",
                load_array("word_embeddings_layernorm.bias"))

    for i in range(config.num_hidden_layers):
        stage_id = i // layers_per_stage
        if stage_id != self.mesh_id:
            continue

        param_prefix = f"params.transformer.h.{i}."
        load_prefix = f"h.{i}."
        # Attention weights
        load_param(param_prefix + "self_attention.query_key_value.kernel",
                   load_array(load_prefix + "self_attention.query_key_value.weight").transpose())
        load_param(param_prefix + "self_attention.query_key_value.bias",
                   load_array(load_prefix + "self_attention.query_key_value.bias").transpose())
        load_param(param_prefix + "input_layernorm.scale",
                   load_array(load_prefix + "input_layernorm.weight"))
        load_param(param_prefix + "input_layernorm.bias",
                   load_array(load_prefix + "input_layernorm.bias"))
        load_param(param_prefix + "self_attention.dense.kernel",
                   load_array(load_prefix + "self_attention.dense.weight"))
        load_param(param_prefix + "self_attention.dense.bias",
                   load_array(load_prefix + "self_attention.dense.bias"))
        load_param(param_prefix + "post_attention_layernorm.scale",
                   load_array(load_prefix + "post_attention_layernorm.weight"))
        load_param(param_prefix + "post_attention_layernorm.bias",
                   load_array(load_prefix + "post_attention_layernorm.bias"))
        # MLP weights
        load_param(param_prefix + "mlp.dense_h_to_4h.kernel",
                   np.transpose(load_array(load_prefix + "mlp.dense_h_to_4h.weight")))
        load_param(param_prefix + "mlp.dense_h_to_4h.bias",
                   np.transpose(load_array(load_prefix + "mlp.dense_h_to_4h.bias")))
        load_param(param_prefix + "mlp.dense_4h_to_h.kernel",
                   np.transpose(load_array(load_prefix + "mlp.dense_4h_to_h.weight")))
        load_param(param_prefix + "mlp.dense_4h_to_h.bias",
                   np.transpose(load_array(load_prefix + "mlp.dense_4h_to_h.bias")))


setattr(MeshHostWorker, "load_bloom_params_worker_func",
        load_bloom_params_worker_func)

def load_params_dis_array(path, executable, params_aval, config, dummy=False):
    """Load parameters with distributed arrays."""
    if dummy:
        alpa.global_config.use_dummy_value_for_benchmarking = True
        params_info, _ = executable.get_input_placement_specs()
        flat_args, in_tree = tree_flatten(params_aval)
        flat_info = tree_leaves(params_info)
        if hasattr(executable, "mesh_group"):
            ret = executable.mesh_group.shard_args_to_arrays(
                flat_info, flat_args)
        else:
            ret = executable.physical_mesh.shard_args_to_arrays_ps(
                flat_info, flat_args)
        alpa.global_config.use_dummy_value_for_benchmarking = False
        return ret

    params_info, _ = executable.get_input_placement_specs()

    prefix_to_flat_idx = {}
    ct = itertools.count()

    def dfs(dict_tree, result_dict, cur_prefix):
        if isinstance(dict_tree, (dict, flax.core.FrozenDict)):
            for key in dict_tree.keys():
                dfs(dict_tree[key], result_dict,
                    cur_prefix + ("." if cur_prefix else "") + key)
        else:
            result_dict[cur_prefix] = next(ct)

    dfs(params_aval, prefix_to_flat_idx, "")

    flat_infos, in_tree = tree_flatten(params_info)

    flat_shapes = []
    flat_uuids = []
    flat_indices = []
    flat_mesh_ids = []
    flat_arrays = []

    mesh_group = executable.mesh_group

    for info in flat_infos:
        aval = info.aval
        if len(info.mesh_ids) == 1:
            mesh, spec = mesh_group[info.mesh_ids[0]], info.sharding_specs[0]
            indices = pxla.spec_to_indices(aval.shape, spec)
            ary_refs, ary_uuid = create_remote_array_refs(mesh)
            flat_shapes.append([aval.shape])
            flat_uuids.append([ary_uuid[0]])
            flat_indices.append([indices])
            flat_mesh_ids.append([mesh.mesh_id])
            flat_arrays.append(
                DistributedArray(mesh, aval, spec, ary_refs[0], indices))
        else:
            tmp_shapes = []
            tmp_uuids = []
            tmp_indices = []
            tmp_mesh_ids = []
            tmp_arrays = []
            tmp_meshes = []
            for mesh_id, spec in zip(info.mesh_ids, info.sharding_specs):
                mesh = mesh_group[mesh_id]
                indices = pxla.spec_to_indices(aval.shape, spec)
                ary_refs, ary_uuid = create_remote_array_refs(mesh)
                array = DistributedArray(mesh, aval, spec, ary_refs[0], indices)
                tmp_shapes.append(aval.shape)
                tmp_uuids.append(ary_uuid[0])
                tmp_indices.append(indices)
                tmp_mesh_ids.append(mesh.mesh_id)
                tmp_meshes.append(mesh)
                tmp_arrays.append(array)
            flat_shapes.append(tuple(tmp_shapes))
            flat_uuids.append(tuple(tmp_uuids))
            flat_indices.append(tuple(tmp_indices))
            flat_mesh_ids.append(tuple(tmp_mesh_ids))
            flat_arrays.append(
                ReplicatedDistributedArray(tmp_meshes, tmp_arrays))

    for m in executable.mesh_group.meshes:
        for w in m.workers:
            w.load_opt_params_worker_func.remote(path, prefix_to_flat_idx,
                                                 config, flat_shapes,
                                                 flat_uuids, flat_indices,
                                                 flat_mesh_ids)

    return flat_arrays


def init_cache_dis_array(executable, config, batch_size, dummy=False):
    """Initialize cache with distributed arrays."""
    cache = init_cache_np(config, batch_size)
    alpa.global_config.use_dummy_value_for_benchmarking = dummy
    _, batch_info = executable.get_input_placement_specs()
    flat_args, in_tree = tree_flatten(cache)
    flat_info = tree_leaves(batch_info["cache"])
    if hasattr(executable, "mesh_group"):
        ret = executable.mesh_group.shard_args_to_arrays(flat_info, flat_args)
    else:
        ret = executable.physical_mesh.shard_args_to_arrays_ps(
            flat_info, flat_args)
    alpa.global_config.use_dummy_value_for_benchmarking = False
    return ret


def load_multi_executable_params_dis_array(path,
                                           executables,
                                           params_aval,
                                           config,
                                           dummy=False):
    """Load parameters to workers that will be used by all executables. Accordingly,
    we need to make sure the parameter sharding specs are identical for all executables.
    """
    shared_input_shard_specs = None
    for executable in executables.values():
        stage_input_shard_specs = executable.stage_input_shard_specs
        if shared_input_shard_specs is not None:
            assert shared_input_shard_specs == stage_input_shard_specs, \
                "All executables must have the same input sharding specs."
        else:
            shared_input_shard_specs = stage_input_shard_specs
    return load_params_dis_array(path,
                                 list(executables.values())[0], params_aval,
                                 config, dummy)


def init_multi_executable_cache_dis_array(executables,
                                          config,
                                          batch_size,
                                          dummy=False):
    """Initialize cache to workers that will be used by all executables. Accordingly,
    we need to make sure all executables are using the same cache.
    """
    cache_info = None
    for executable in executables.values():
        _, batch_info = executable.get_input_placement_specs()
        if cache_info is not None:
            assert cache_info == batch_info["cache"], \
                "All executables must share the same cache"
        else:
            cache_info = batch_info["cache"]
    return init_cache_dis_array(
        list(executables.values())[0], config, batch_size, dummy)