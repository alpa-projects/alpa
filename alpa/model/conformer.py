"""Conformer.

Reference:
https://arxiv.org/pdf/2005.08100.pdf
https://github.com/TensorSpeech/TensorFlowASR/blob/main/tensorflow_asr/models/encoders/conformer.py
"""

from functools import partial
from typing import Any, Callable

import numpy as np

import flax
from flax import linen as nn, optim
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp

from alpa.model.model_util import (FlaxBaseModelOutput,
                                   FlaxBaseModelOutputWithPooling,
                                   FlaxBertForPreTrainingOutput,
                                   FlaxMaskedLMOutput)
from alpa import mark_pipeline


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: optim.DynamicScale


class ConformerConfig:

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 gradient_checkpointing=False,
                 position_embedding_type="absolute",
                 use_cache=True,
                 conv_subsample_channel=256,
                 conv_kernel_size=32,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.conv_subsample_channel = conv_subsample_channel
        self.conv_kernel_size = conv_kernel_size


class ConvSubSample(nn.Module):
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv1 = nn.Conv(features=self.config.conv_subsample_channel,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             dtype=self.dtype)
        self.conv2 = nn.Conv(features=self.config.conv_subsample_channel,
                             kernel_size=(3, 3),
                             strides=(2, 2),
                             dtype=self.dtype)
        self.dense = nn.Dense(features=self.config.hidden_size,
                              dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, x, deterministic: bool = True):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.dense(x)
        x = self.dropout(x, deterministic=deterministic)
        return x


class FFNModule(nn.Module):
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)
        self.dense_1 = nn.Dense(self.config.intermediate_size, dtype=self.dtype)
        self.act = nn.swish
        self.dropout_1 = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.dense_2 = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.dropout_2 = nn.Dropout(rate=self.config.hidden_dropout_prob)

    def __call__(self, inputs, deterministic: bool = True):
        outputs = self.layer_norm(inputs)
        outputs = self.dense_1(outputs)
        outputs = self.act(outputs)
        outputs = self.dropout_1(outputs, deterministic=deterministic)
        outputs = self.dense_2(outputs)
        outputs = self.dropout_2(outputs, deterministic=deterministic)
        return 0.5 * outputs + inputs


class ConvModule(nn.Module):
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, deterministic: bool = True, train: bool = True):
        outputs = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                               dtype=self.dtype)(inputs)
        B, T, E = outputs.shape
        outputs = outputs.reshape((B, T, 1, E))
        outputs = nn.Conv(features=self.config.hidden_size * 2,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          dtype=self.dtype)(outputs)
        outputs = nn.glu(outputs)
        outputs = nn.Conv(features=self.config.hidden_size,
                          kernel_size=(self.config.conv_kernel_size, 1),
                          strides=(1, 1),
                          feature_group_count=self.config.hidden_size,
                          dtype=self.dtype)(outputs)
        outputs = nn.BatchNorm(use_running_average=not train,
                               momentum=0.9,
                               epsilon=1e-5,
                               dtype=self.dtype)(outputs)
        outputs = nn.swish(outputs)
        outputs = nn.Conv(features=self.config.hidden_size,
                          kernel_size=(1, 1),
                          strides=(1, 1),
                          dtype=self.dtype)(outputs)
        outputs = outputs.reshape((B, T, E))
        outputs = nn.Dropout(rate=self.config.hidden_dropout_prob)(
            outputs, deterministic=deterministic)
        return outputs + inputs


class MultiHeadSelfAttentionModule(nn.Module):
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)
        self.qvk_combined = nn.Dense(
            self.config.hidden_size * 3,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.out_dense = nn.Dense(self.config.hidden_size,
                                  dtype=self.dtype,
                                  kernel_init=jax.nn.initializers.normal(
                                      self.config.initializer_range))

        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size`: {self.config.hidden_size} has to be a multiple of `num_attention_heads`: {self.config.num_attention_heads}"
            )

    def __call__(self,
                 inputs,
                 pos_encoding,
                 attention_mask,
                 deterministic=True):
        outputs = self.layer_norm(inputs)
        outputs = outputs + pos_encoding

        head_dim = self.config.hidden_size // self.config.num_attention_heads

        qvk_combined_states = self.qvk_combined(outputs)
        qvk_combined_states = qvk_combined_states.reshape(
            qvk_combined_states.shape[:2] + (-1, 3))
        query_states, value_states, key_states = jnp.split(qvk_combined_states,
                                                           3,
                                                           axis=3)
        query_states = query_states.reshape(outputs.shape[:2] +
                                            (self.config.num_attention_heads,
                                             head_dim))
        value_states = value_states.reshape(outputs.shape[:2] +
                                            (self.config.num_attention_heads,
                                             head_dim))
        key_states = key_states.reshape(outputs.shape[:2] +
                                        (self.config.num_attention_heads,
                                         head_dim))

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None
        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights,
                                 value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = self.out_dense(attn_output)
        outputs = self.dropout(outputs, deterministic=deterministic)
        return outputs + inputs


class ConformerLayer(nn.Module):
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.ffn_1 = FFNModule(config=self.config, dtype=self.dtype)
        self.mhsa = MultiHeadSelfAttentionModule(config=self.config,
                                                 dtype=self.dtype)
        self.conv = ConvModule(config=self.config, dtype=self.dtype)
        self.ffn_2 = FFNModule(config=self.config, dtype=self.dtype)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(
        self,
        inputs,
        pos_encoding,
        attention_mask,
        deterministic: bool = True,
        train: bool = True,
    ):
        outputs = self.ffn_1(inputs, deterministic=deterministic)
        outputs = self.mhsa(outputs,
                            pos_encoding,
                            attention_mask,
                            deterministic=deterministic)
        outputs = self.conv(outputs, deterministic=deterministic, train=train)
        outputs = self.ffn_2(outputs, deterministic=deterministic)
        outputs = self.layer_norm(outputs)
        return outputs


class ConformerForASRModule(nn.Module):
    """
    Conformer for automatic speech recognition.
    """
    config: ConformerConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv_subsample = ConvSubSample(config=self.config,
                                            dtype=self.dtype)
        self.layers = [
            ConformerLayer(config=self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]
        self.decoder = nn.Dense(self.config.vocab_size, dtype=self.dtype)

    def __call__(
        self,
        input_frames,
        attention_mask,
        deterministic: bool = True,
        train: bool = True,
    ):
        # Model
        hidden_states = self.conv_subsample(input_frames)
        pos_encoding = jnp.ones(
            (1, hidden_states.shape[1], hidden_states.shape[2]))

        for layer in self.layers:
            hidden_states = layer(hidden_states,
                                  pos_encoding,
                                  attention_mask,
                                  deterministic=deterministic,
                                  train=train)

        logits = self.decoder(hidden_states)

        return logits
