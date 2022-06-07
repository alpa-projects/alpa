# flake8: noqa
"""Model definition of Mixture of Expert model."""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np

import flax
from flax import optim, linen as nn
from flax.training import train_state
from flax.linen.attention import dot_product_attention_weights
from flax.linen.initializers import lecun_normal
import jax
from jax import lax
import jax.numpy as jnp
from jax.nn import one_hot

from alpa.model.bert_model import (FlaxBaseModelOutput,
                                   FlaxBaseModelOutputWithPooling,
                                   FlaxBertAttention, FlaxBertEmbeddings,
                                   FlaxBertIntermediate, FlaxBertLayer,
                                   FlaxBertOutput, FlaxMaskedLMOutput)
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary


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
            add_manual_pipeline_markers=False,
            pipeline_mp_size=0,
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
        self.expert_group_size = expert_group_size
        self.expert_number = expert_number
        self.tie_word_embeddings = tie_word_embeddings
        self.add_manual_pipeline_markers = add_manual_pipeline_markers
        self.pipeline_mp_size = pipeline_mp_size


def top2_gating_dummy(gates):  # [GSE] -> [GSEC, GSEC]
    """A temporary dummy implementation."""
    G, S, E = gates.shape
    C = 2 * S // E
    gates = jnp.reshape(gates, (G, S, E, 1))
    combined_weights = jnp.broadcast_to(gates, (G, S, E, C))
    dispatch_mask = combined_weights
    return combined_weights, dispatch_mask


def top2_gating(gates):  # GSE -> (GSEC, GSEC)
    """Modified from https://github.com/tensorflow/lingvo/blob/
    b885b91d4b5361c971a998b810fc58f83baa625f/lingvo/core/gshard_layers.py#L1787

    # TODO(lmzheng): add the auxiliary loss. add 'random' policy for the second expert.
    """
    G, S, E = gates.shape
    C = 2 * S // E

    mask_dtype = jnp.int32

    index_1 = jnp.argmax(gates, axis=-1)  # GS
    mask_1 = one_hot(index_1, E, dtype=mask_dtype)  # GSE
    gate_1 = jnp.einsum("GSE,GSE->GS", gates, mask_1)  # GS

    gates_without_top_1 = gates * (1 - mask_1)

    index_2 = jnp.argmax(gates_without_top_1, axis=-1)  # GSE
    mask_2 = one_hot(index_2, E, dtype=mask_dtype)
    gate_2 = jnp.einsum("GSE,GSE->GS", gates_without_top_1, mask_2)

    pos_1 = jnp.cumsum(mask_1, axis=-2) - mask_1
    mask_1 *= pos_1 < C
    pos_1 = jnp.einsum("GSE,GSE->GS", pos_1, mask_1)

    mask_1_count = jnp.sum(mask_1, axis=-2)
    mask_1_flat = jnp.sum(mask_1, axis=-1)

    pos_2 = (jnp.cumsum(mask_2, axis=-2) - mask_2) + jnp.expand_dims(
        mask_1_count, -2)
    mask_2 *= pos_2 < C
    pos_2 = jnp.einsum("GSE,GSE->GS", pos_2, mask_2)

    mask_2_flat = jnp.sum(mask_2, axis=-1)

    gate_1 *= mask_1_flat
    gate_2 *= mask_2_flat

    denom = gate_1 + gate_2
    denom = jnp.where(denom > 0, denom, jnp.ones_like(denom))
    gate_1 /= denom
    gate_2 /= denom

    a = jnp.expand_dims(gate_1 * mask_1_flat, -1) * one_hot(
        index_1, E, dtype=gates.dtype)
    b = one_hot(pos_1, C, dtype=gates.dtype)
    first_part_of_combine_tensor = jnp.einsum("GSE,GSC->GSEC", a, b)

    a = jnp.expand_dims(gate_2 * mask_2_flat, -1) * one_hot(
        index_2, E, dtype=gates.dtype)
    b = one_hot(pos_2, C, dtype=gates.dtype)
    second_part_of_combine_tensor = jnp.einsum("GSE,GSC->GSEC", a, b)

    combined_tensor = first_part_of_combine_tensor + second_part_of_combine_tensor
    dispatch_tensor = combined_tensor.astype(jnp.bool_)

    return combined_tensor, dispatch_tensor


class FlaxPositionWiseMoELayer(nn.Module):
    config: MoEConfig
    kernel_init: Callable[..., np.ndarray] = lecun_normal()
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(self, inputs):
        S = self.config.expert_group_size
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
        gates = jax.nn.softmax(jnp.einsum("GSM,ME->GSE", reshaped_inputs, wg))
        combined_weights, dispatch_mask = top2_gating(gates)
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

        if not isinstance(deterministic, bool):
            # A temporary hack to walkaround the bug in flax.nn.remat
            # Using `nn.remat(concrete=True)` works for regular use cases
            # (e.g., train_step, init) but does not work for init_dummy.
            # So we still need this hack.
            deterministic = True
            output_attentions = True

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

        if self.config.gradient_checkpointing:
            trans_func = partial(nn.remat, concrete=True)
        else:
            trans_func = lambda x: x

        assert self.config.num_hidden_layers % 2 == 0
        layers = []
        for i in range(self.config.num_hidden_layers):
            if i % 2 == 0:
                layers.append(
                    trans_func(FlaxMoELayer)(self.config,
                                             name=str(i),
                                             dtype=self.dtype))
            else:
                layers.append(
                    trans_func(FlaxBertLayer)(self.config,
                                              name=str(i),
                                              dtype=self.dtype))
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

            if self.config.add_manual_pipeline_markers:
                layers_per_stage = self.config.num_hidden_layers // self.config.pipeline_mp_size
                assert self.config.num_hidden_layers % self.config.pipeline_mp_size == 0
                if i % layers_per_stage == layers_per_stage - 1 and i != len(
                        self.layers) - 1:
                    mark_pipeline_boundary()

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
