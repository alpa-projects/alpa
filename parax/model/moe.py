# flake8: noqa
"""Model definition of Mixture of Expert model."""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

import flax
from flax import optim
from flax.linen.attention import dot_product_attention_weights
from flax.linen.initializers import lecun_normal
import flax.linen as nn
import jax
from jax import lax
import jax.numpy as jnp

from parax.model.bert_model import (FlaxBaseModelOutput,
                                    FlaxBaseModelOutputWithPooling,
                                    FlaxBertAttention, FlaxBertEmbeddings,
                                    FlaxBertIntermediate, FlaxBertLayer,
                                    FlaxBertOutput, FlaxMaskedLMOutput)


class MoEConfig:

    def __init__(
            self,
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            gradient_checkpointing=False,
            position_embedding_type="absolute",
            use_cache=True,
            tie_word_embeddings=True,
            expert_group_size=8192,  # S in the paper
            expert_number=128,  # E in the paper
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
        self.tie_word_embeddings = tie_word_embeddings
        self.expert_group_size = expert_group_size
        self.expert_number = expert_number


def top2gating(gates):  # [GSE] -> [GSEC, GSEC]
    """A temporary dummy implementation."""
    G, S, E = gates.shape
    C = 2 * S // E
    gates = jnp.reshape(gates, (G, S, E, 1))
    combined_weights = jnp.broadcast_to(gates, (G, S, E, C))
    dispatch_mask = combined_weights
    return combined_weights, dispatch_mask


class FlaxPositionWiseMoELayer(nn.Module):
    config: MoEConfig
    kernel_init: Callable[..., np.ndarray] = lecun_normal()
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs):
        S = self.config.expert_group_size
        M = self.config.hidden_size
        M = self.config.hidden_size
        H = self.config.intermediate_size
        E = self.config.expert_number

        wg = self.param("wg", self.kernel_init, (
            M,
            E,
        ))
        wi = self.param("wi", self.kernel_init, (
            E,
            M,
            H,
        ))
        wo = self.param("wo", self.kernel_init, (
            E,
            H,
            M,
        ))

        inputs = jnp.asarray(inputs, self.dtype)
        wg = jnp.asarray(wg, self.dtype)
        wi = jnp.asarray(wi, self.dtype)
        wo = jnp.asarray(wo, self.dtype)

        reshaped_inputs = jnp.reshape(inputs, (-1, S, M))
        gates = jnp.einsum("GSM,ME->GSE", reshaped_inputs, wg)
        combined_weights, dispatch_mask = top2gating(gates)
        dispatched_expert_inputs = jnp.einsum("GSEC,GSM->EGCM", dispatch_mask,
                                              reshaped_inputs)
        h = jnp.einsum("EGCM,EMH->EGCH", dispatched_expert_inputs, wi)
        h = nn.relu(h)
        expert_outputs = jnp.einsum("EGCH,EHM->GECM", h, wo)
        outputs = jnp.einsum("GSEC,GECM->GSM", combined_weights, expert_outputs)
        outputs = jnp.reshape(outputs, inputs.shape)
        return outputs


class FlaxMoELayer(nn.Module):
    config: MoEConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBertAttention(self.config, dtype=self.dtype)
        self.moe = FlaxPositionWiseMoELayer(self.config, dtype=self.dtype)
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                      dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic: bool = True,
                 output_attentions: bool = False):
        attention_outputs = self.attention(hidden_states,
                                           attention_mask,
                                           deterministic=deterministic,
                                           output_attentions=output_attentions)
        attention_output = attention_outputs[0]

        hidden_states = self.moe(attention_output)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxMoELayerCollection(nn.Module):
    config: MoEConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        assert self.config.num_hidden_layers % 2 == 0

        layers = []
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 0:
                layers.append(
                    FlaxMoELayer(self.config, name=str(i), dtype=self.dtype))
            else:
                layers.append(
                    FlaxBertLayer(self.config, name=str(i), dtype=self.dtype))
        self.layers = layers

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states,
                                  attention_mask,
                                  deterministic=deterministic,
                                  output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(last_hidden_state=hidden_states,
                                   hidden_states=all_hidden_states,
                                   attentions=all_attentions)


class FlaxMoEEncoder(nn.Module):
    config: MoEConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxMoELayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxMoEModule(nn.Module):
    config: MoEConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxMoEEncoder(self.config, dtype=self.dtype)
        if self.add_pooling_layer:
            self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(input_ids,
                                        token_type_ids,
                                        position_ids,
                                        attention_mask,
                                        deterministic=deterministic)
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxMoEForLMModule(nn.Module):
    config: MoEConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = FlaxMoEModule(config=self.config,
                                          add_pooling_layer=False,
                                          dtype=self.dtype)

        if self.config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = nn.Dense(self.config.vocab_size,
                                    dtype=self.dtype,
                                    use_bias=False)
        self.decoder_bias = self.param("bias", self.bias_init,
                                       (self.config.vocab_size,))

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.transformers(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            shared_embedding = self.transformers.variables["params"][
                "embeddings"]["word_embeddings"]["embedding"]
            assert self.decoder is None
            logits = hidden_states @ shared_embedding.T
        else:
            assert self.decoder is not None
            logits = self.decoder(hidden_states)

        logits += jnp.asarray(self.decoder_bias, self.dtype)

        # Compute the prediction scores
        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
