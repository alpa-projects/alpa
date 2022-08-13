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
""" Flax BLOOM model."""
# TODO: see todos throughout this file
# TODO: check correctness against pytorch implementation
# TODO: add unit tests
# TODO: add documentation / check that documentation is correct
# TODO: BLOOM_INPUTS_DOCSTRING might be wrong still (position_ids)
# TODO: check that code is jit-able

import math
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
from flax.linen.activation import tanh
from flax.linen.partitioning import scan_with_axes
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers.modeling_flax_outputs import FlaxCausalLMOutput, FlaxBaseModelOutputWithPast
from transformers.utils import logging

from .configuration_bloom import BloomConfig
from .modeling_flax_utils import FlaxPreTrainedModel
from . import layers
from .layers import with_sharding_constraint


logger = logging.get_logger(__name__)


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

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=n_head, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_flax_t5.py#L426
    # batch_size = 1, n_head = n_head, query_length
    batch_size, key_length = attention_mask.shape
    num_heads = n_head
    query_length = 1

    slopes = jnp.array(get_slopes(n_head))[None, :, None, None].astype(dtype)
    arange_tensor = attention_mask.cumsum(-1, dtype=dtype)[:, None, None, :] - 1

    slopes_broadcasted = jnp.broadcast_to(slopes, (batch_size, num_heads, query_length, key_length))
    arange_broadcasted = jnp.broadcast_to(arange_tensor, (batch_size, num_heads, query_length, key_length))

    alibi = slopes_broadcasted * arange_broadcasted
    return alibi


class FlaxBloomAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size
        self.num_heads = self.config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.attention_softmax_in_fp32 = self.config.attention_softmax_in_fp32

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by `num_heads` (got `hidden_size`: {self.hidden_size} and "
                f"`num_heads`: {self.num_heads})."
            )

        self.query_key_value = layers.DenseGeneral(
            axis=-1,
            features=(self.num_heads, self.head_dim * 3),
            kernel_axes=('embed', 'heads', 'kv'),
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.dense = layers.DenseGeneral(
            features=self.hidden_size,
            axis=(-2, -1),
            kernel_axes=('heads', 'kv', 'embed'),
            dtype=self.dtype,
            params_dtype=self.params_dtype,
        )
        self.resid_dropout = nn.Dropout(rate=self.config.hidden_dropout)

    @nn.compact
    def _concatenate_to_cache(self, key, value, query, attention_mask):
        # The following code is largely copied from:
        # https://github.com/google-research/t5x/blob/63d9addf628c6d8c547a407a32095fcb527bb20b/t5x/examples/scalable_t5/layers.py#L280-L284
        is_initialized = self.has_variable('cache', 'cached_key')
        # The key and value have dimension [batch_size, seq_length, num_heads, head_dim],
        # but we cache them as [batch_size, num_heads, head_dim, seq_length] as a TPU
        # fusion optimization. This also enables the "scatter via one-hot
        # broadcast" trick, which means we do a one-hot broadcast instead of a
        # scatter/gather operations, resulting in a 3-4x speedup in practice.
        swap_dims = lambda x: x[:-3] + tuple(x[i] for i in [-2, -1, -3])  # noqa: E731

        cached_key = self.variable('cache', 'cached_key', jnp.zeros, swap_dims(key.shape), key.dtype)
        cached_value = self.variable('cache', 'cached_value', jnp.zeros, swap_dims(value.shape), value.dtype)
        cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))

        if is_initialized:
            batch_size, num_heads, head_dim, seq_length = (cached_key.value.shape)
            # During fast autoregressive decoding, we feed one position at a time,
            # and cache the keys and values step by step.
            # Sanity shape check of cached key against input query.
            num_updated_cache_vectors = query.shape[1]
            expected_shape = (batch_size, 1, num_heads, head_dim)
            if num_updated_cache_vectors == 1 and expected_shape != query.shape:
                raise ValueError(f"Autoregressive cache shape error, expected query shape {expected_shape} instead got {query.shape}")

            # Create a OHE of the current index. NOTE: the index is increased below.
            cur_index = cache_index.value

            # In order to update the key, value caches with the current key and
            # value, we move the seq_length axis to the back, similar to what we did for
            # the cached ones above.
            # Note these are currently the key and value of a single position, since
            # we feed one position at a time.
            one_token_key = jnp.moveaxis(key, -3, -1)
            one_token_value = jnp.moveaxis(value, -3, -1)

            # Update key, value caches with our new 1d spatial slices.
            # We implement an efficient scatter into the cache via one-hot
            # broadcast and addition.
            if num_updated_cache_vectors > 1:
                indices = jnp.eye(num_updated_cache_vectors, seq_length)[None, None]
                key = cached_key.value + jnp.matmul(one_token_key, indices)
                value = cached_value.value + jnp.matmul(one_token_value, indices)
            else:
                one_hot_indices = jax.nn.one_hot(cur_index, seq_length, dtype=key.dtype)
                key = cached_key.value + one_token_key * one_hot_indices
                value = cached_value.value + one_token_value * one_hot_indices

            cached_key.value = key
            cached_value.value = value
            cache_index.value = cache_index.value + num_updated_cache_vectors

            # Move the keys and values back to their original shapes.
            key = jnp.moveaxis(key, -1, -3)
            value = jnp.moveaxis(value, -1, -3)

            # causal mask for cached decoder self-attention: our single query position should only attend to those key positions that have already been generated and cached, not the remaining zero elements.
            pad_mask = jnp.broadcast_to(
                jnp.arange(seq_length) < cur_index + num_updated_cache_vectors,
                (batch_size,) + (1, num_updated_cache_vectors, seq_length),
            )
            attention_mask = combine_masks(pad_mask, attention_mask)
        return key, value, attention_mask

    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        layer_number: int = None,
    ):  

        batch_size, seq_length = hidden_states.shape[:2]

        # proj q, k, v
        fused_qkv = self.query_key_value(hidden_states)
        query, key, value = jnp.split(fused_qkv, 3, axis=-1)
        
        query = with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
        key = with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
        value = with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

        causal_attention_mask = make_causal_mask(attention_mask, dtype="bool")

        # for fast decoding causal attention mask should be shifted
        causal_attention_mask_shift = (
            self.variables["cache"]["cache_index"] if self.has_variable("cache", "cached_key") else 0
        )

        # fast decoding for generate requires special attention_mask
        if self.has_variable("cache", "cached_key"):
            # sequence_length of cached_key is last dim
            # TODO(PVP) - uncomment other three
            max_decoder_length = self.variables["cache"]["cached_key"].shape[-1]
            causal_attention_mask = jax.lax.dynamic_slice(
                causal_attention_mask,
                (0, 0, causal_attention_mask_shift, 0),
                (1, 1, seq_length, max_decoder_length),
            )

        # broadcast causal attention mask & attention mask to fit for merge
        causal_attention_mask = jnp.broadcast_to(
            causal_attention_mask, (batch_size,) + causal_attention_mask.shape[1:]
        )
        attention_mask = jnp.broadcast_to(jnp.expand_dims(attention_mask, axis=(-3, -2)), causal_attention_mask.shape)

        attention_mask = combine_masks(attention_mask, causal_attention_mask)

        dropout_rng = None
        deterministic = True
        if not deterministic and self.config.attention_dropout > 0.0:
            dropout_rng = self.make_rng("dropout")

        # During fast autoregressive decoding, we feed one position at a time,
        # and cache the keys and values step by step.
        if self.has_variable("cache", "cached_key") or init_cache:
            key, value, attention_mask = self._concatenate_to_cache(key, value, query, attention_mask)

        # transform boolean mask into float mask
        mask_value = jnp.finfo(self.dtype).min
        attention_bias = lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
            jnp.full(attention_mask.shape, mask_value).astype(self.dtype),
        )

        attention_bias = attention_bias + alibi

        # TODO: override softmax precision to fp32 if self.attention_softmax_in_fp32=True and self.dtype != fp32
        # usual dot product attention
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
        attn_output = self.dense(attn_output)

        outputs = (attn_output, attn_weights) if output_attentions else (attn_output,)
        return outputs


class BloomGELU(nn.Module):
    def setup(self):
        self.dtype = jnp.float32

    def __call__(self, x):
        return x * 0.5 * (1.0 + tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


class FlaxBloomMLP(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.hidden_size = self.config.hidden_size

        self.pretraining_tp = self.config.pretraining_tp
        self.slow_but_exact = self.config.slow_but_exact


        self.dense_h_to_4h = layers.DenseGeneral(
            features=self.hidden_size * 4,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=('embed', 'mlp'),
        )
        self.dense_4h_to_h = layers.DenseGeneral(
            features=self.hidden_size,
            dtype=self.dtype,
            params_dtype=self.params_dtype,
            kernel_axes=('mlp', 'embed'),
        )
        self.hidden_dropout = nn.Dropout(self.config.hidden_dropout)
        self.act = BloomGELU()

    def __call__(self, hidden_states, deterministic: bool = True):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)

        # TODO: this code block is from the pytorch implementation. needs changing to work.
        #        if self.pretraining_tp > 1 and self.slow_but_exact:
        # if False:
        #     intermediate_output = jnp.zeros_like(residual)
        #     slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
        #     for i in range(self.pretraining_tp):
        #         intermediate_output = intermediate_output + nn.functional.linear(
        #             hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
        #             self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
        #         )
        # else:
        #     intermediate_output = self.dense_4h_to_h(hidden_states)

        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states


class FlaxBloomBlock(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        self.input_layernorm = layers.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype, params_dtype=self.params_dtype)

        self.self_attention = FlaxBloomAttention(self.config, dtype=self.dtype, params_dtype=self.params_dtype)
        self.post_attention_layernorm = layers.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype, params_dtype=self.params_dtype)

        self.mlp = FlaxBloomMLP(self.config, dtype=self.dtype, params_dtype=self.params_dtype)

        self.apply_residual_connection_post_layernorm = self.config.apply_residual_connection_post_layernorm
        self.hidden_dropout = self.config.hidden_dropout

    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        layer_number: int = None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
    ):
        if self.use_scan:
            hidden_states = hidden_states[0]

        hidden_states = with_sharding_constraint(hidden_states, ('batch', 'length', 'embed'))
        layernorm_output = self.input_layernorm(hidden_states)
        layernorm_output = with_sharding_constraint(layernorm_output, ('batch', 'length', 'embed'))
        
        # layer norm before saving residual if config calls for it
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # self-attention
        attn_outputs = self.self_attention(
            layernorm_output,
            alibi=alibi,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_attentions=output_attentions,
            layer_number=layer_number,
        )

        attention_output = attn_outputs[0]
        # apply residual connection
        attention_output = attention_output + residual
        attention_output = with_sharding_constraint(attention_output, ('batch', 'length', 'embed'))

        outputs = attn_outputs[1:]

        post_layernorm = self.post_attention_layernorm(attention_output)
        post_layernorm = with_sharding_constraint(post_layernorm, ('batch', 'length', 'embed'))

        # set residual based on config
        if self.apply_residual_connection_post_layernorm:
            residual = post_layernorm
        else:
            residual = attention_output

        output = self.mlp(post_layernorm, deterministic=deterministic)
        output = output + residual
        output = with_sharding_constraint(output, ('batch', 'length', 'embed'))

        outputs = (output,) + outputs[1:]

        if self.use_scan:
            outputs = (outputs, None)

        return outputs


# TODO: does this still require position_ids?
# TODO: gradient checkpointing
# TODO: _no_split_modules?
# TODO: check initialization
class FlaxBloomPreTrainedModel(FlaxPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    module_class: nn.Module = None

    def __init__(
        self,
        config: BloomConfig,
        input_shape: Tuple = (1, 1),
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        params_dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
        use_scan: bool = False,
        **kwargs,
    ):
        module = self.module_class(config=config, dtype=dtype, use_scan=use_scan, params_dtype=params_dtype, **kwargs)
        super().__init__(config, module, input_shape=input_shape, seed=seed, dtype=dtype, _do_init=_do_init)

    def init_weights(self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict = None) -> FrozenDict:
        # init input tensors
        input_ids = jnp.zeros(input_shape, dtype="i4")
        attention_mask = jnp.ones_like(input_ids)
        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        random_params = self.module.init(rngs, input_ids, attention_mask, return_dict=False)["params"]

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def init_cache(self, batch_size, max_length):
        r"""
        Args:
            batch_size (`int`):
                batch_size used for fast auto-regressive decoding. Defines the batch size of the initialized cache.
            max_length (`int`):
                maximum possible length for auto-regressive decoding. Defines the sequence length of the initialized
                cache.
        """
        # init input variables to retrieve cache
        input_ids = jnp.ones((batch_size, max_length))
        attention_mask = jnp.ones_like(input_ids)

        init_variables = self.module.init(
            jax.random.PRNGKey(0), input_ids, attention_mask, return_dict=False, init_cache=True
        )
        return unfreeze(init_variables["cache"])

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        past_key_values: dict = None,
        params: dict = None,
        dropout_rng: jax.random.PRNGKey = None,
        train: bool = False,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, sequence_length = input_ids.shape

        if attention_mask is None:
            attention_mask = jnp.ones((batch_size, sequence_length))

        # Handle any PRNG if needed
        rngs = {}
        if dropout_rng is not None:
            rngs["dropout"] = dropout_rng

        inputs = {"params": params or self.params}

        # if past_key_values are passed then cache is already initialized a private flag init_cache has to be passed down to ensure cache is used. It has to be made sure that cache is marked as mutable so that it can be changed by FlaxBloomAttention module
        if past_key_values:
            inputs["cache"] = past_key_values
            mutable = ["cache"]
        else:
            mutable = False

        outputs = self.module.apply(
            inputs,
            jnp.array(input_ids, dtype="i4"),
            jnp.array(attention_mask, dtype="i4"),
            not train,
            False,
            output_attentions,
            output_hidden_states,
            return_dict,
            rngs=rngs,
            mutable=mutable,
        )

        # add updated cache to model output
        if past_key_values is not None and return_dict:
            outputs, past_key_values = outputs
            outputs["past_key_values"] = unfreeze(past_key_values["cache"])
            return outputs
        elif past_key_values is not None and not return_dict:
            outputs, past_key_values = outputs
            outputs = outputs[:1] + (unfreeze(past_key_values["cache"]),) + outputs[1:]

        return outputs


class FlaxBloomBlockCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    # TODO (SG): re-write as a `setup` to conform to Transformers JAX/Flax conventions -> awaiting CG response on G Chat
    @nn.compact
    def __call__(
        self,
        hidden_states,
        alibi,
        attention_mask=None,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if self.use_scan:
            # since all decoder layers are the same, we use nn.scan directly
            # assert not output_attentions, "cannot use `scan` with `output_attentions` set to `True`"
            # assert not output_hidden_states, "cannot use `scan` with `output_hidden_states` set to `True`"
            hidden_states = (hidden_states,)
            layer_number = jnp.arange(self.config.num_hidden_layers)

            hidden_states, _ = scan_with_axes(
                FlaxBloomBlock,
                variable_axes={"params": 0, "cache": 0},
                split_rngs={"params": True, "dropout": True},
                in_axes=(nn.broadcast, nn.broadcast, 0, nn.broadcast, nn.broadcast),
                length=self.config.num_hidden_layers,
            )(self.config, dtype=self.dtype, params_dtype=self.params_dtype, use_scan=True, name="FlaxBloomBlockLayers")(
                hidden_states,
                alibi,
                attention_mask,  # kwargs not supported by scan
                layer_number,
                deterministic,
                init_cache,
            )
            hidden_states = hidden_states[0]

        else:
            for layer_number in range(self.config.num_hidden_layers):
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                layer_outputs = FlaxBloomBlock(self.config, name=str(layer_number), dtype=self.dtype, params_dtype=self.params_dtype, use_scan=False)(
                    hidden_states,
                    alibi=alibi,
                    attention_mask=attention_mask,
                    deterministic=deterministic,
                    init_cache=init_cache,
                    output_attentions=output_attentions,
                    layer_number=layer_number,
                )
                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions += (layer_outputs[1],)

        # this contains possible `None` values - `FlaxBloomModule` will filter them out
        outputs = (hidden_states, all_hidden_states, all_attentions)

        return outputs


class FlaxBloomModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        # TODO: check initialization correctness
        self.embed_dim = self.config.hidden_size

        self.embedding_init = jax.nn.initializers.normal(stddev=self.config.initializer_range, dtype=self.params_dtype)

        self.word_embeddings = layers.Embed(
            self.config.vocab_size,
            self.embed_dim,
            embedding_init=nn.initializers.zeros,
            params_dtype=self.params_dtype,
        )
        # post-embedding layernorm
        self.word_embeddings_layernorm = layers.LayerNorm(epsilon=self.config.layer_norm_epsilon, params_dtype=self.params_dtype)

        # transformer layers
        self.h = FlaxBloomBlockCollection(self.config, dtype=self.dtype, params_dtype=self.params_dtype, use_scan=self.use_scan)

        # final layernorm
        self.ln_f = layers.LayerNorm(epsilon=self.config.layer_norm_epsilon, dtype=self.dtype, params_dtype=self.params_dtype)
        # TODO: change how gradient checkpointing is done
        self.gradient_checkpointing = False

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        deterministic=True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        # do post-embedding layernorm
        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        # build alibi depending on `attention_mask`
        alibi = build_alibi_tensor_flax(attention_mask, self.config.n_head, hidden_states.dtype)

        # TODO: gradient checkpointing
        outputs = self.h(
            hidden_states,
            alibi=alibi,
            attention_mask=attention_mask,
            deterministic=deterministic,
            init_cache=init_cache,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        hidden_states = outputs[0]
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs[1] + (hidden_states,)
            outputs = (hidden_states, all_hidden_states) + outputs[2:]
        else:
            outputs = (hidden_states,) + outputs[1:]

        if not return_dict:
            # TODO: don't think this return value / ordering is correct
            return tuple(v for v in [outputs[0], outputs[-1]] if v is not None)

        return FlaxBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=outputs[1],
            attentions=outputs[-1],
        )


class FlaxBloomModel(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomModule


class FlaxBloomForCausalLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float32
    params_dtype: jnp.dtype = jnp.float32
    use_scan: bool = False

    def setup(self):
        self.transformer = FlaxBloomModule(self.config, dtype=self.dtype, params_dtype=self.params_dtype, use_scan=self.use_scan)
        self.lm_head = layers.DenseGeneral(
            self.config.vocab_size,
            dtype=jnp.float32,  # Use float32 for stabiliity.
            params_dtype=self.params_dtype,
            use_bias=False,
            kernel_axes=('embed', 'vocab'),
        )

    def __call__(
        self,
        input_ids,
        attention_mask,
        deterministic: bool = True,
        init_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
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

        return FlaxCausalLMOutput(logits=lm_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


class FlaxBloomForCausalLM(FlaxBloomPreTrainedModel):
    module_class = FlaxBloomForCausalLMModule

    # TODO: check if this class is correct / take out position ids
    def prepare_inputs_for_generation(self, input_ids, max_length, attention_mask: Optional[jnp.DeviceArray] = None):
        # initializing the cache
        batch_size, seq_length = input_ids.shape

        past_key_values = self.init_cache(batch_size, max_length)
        # Note that usually one would have to put 0's in the attention_mask for x > input_ids.shape[-1] and x < cache_length.
        # But since Bloom uses a causal mask, those positions are masked anyways.
        # Thus we can create a single static attention_mask here, which is more efficient for compilation
        extended_attention_mask = jnp.ones((batch_size, max_length), dtype="i4")
        if attention_mask is not None:
            extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))

        return {
            "past_key_values": past_key_values,
            "attention_mask": extended_attention_mask,
        }

    def update_inputs_for_generation(self, model_outputs, model_kwargs):
        model_kwargs["past_key_values"] = model_outputs.past_key_values
        return model_kwargs
