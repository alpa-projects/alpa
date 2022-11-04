import heapq
import math
import queue
import time
import logging

import torch
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Union

import cupy
import cupyx.jit
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
import numpy as np
import os
from enum import Enum
from functools import partial

from alpa.collective.worker_nccl_util_cupy import jax_tensor_to_cupy
from alpa.model.model_util import ModelOutput
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.util import OrderedSet

try:
    from ft_mha import fused_mmha
except ImportError:
    raise RuntimeError("Please install ft_mha to use 1D OPT model.")


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
}


@flax.struct.dataclass
class OPTModelOutput(ModelOutput):
    last_hidden_state: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@flax.struct.dataclass
class OPTLMOutput(ModelOutput):
    logits: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@dataclass(frozen=True)
class OPTConfig:
    # Inherited from OPT
    num_hidden_layers: int = 12
    max_seq_len: int = 2048
    hidden_size: int = 768
    n_head: int = 12
    input_dim: int = 768
    ffn_embed_dim: int = 3072
    pad: int = 1
    activation_fn: str = 'relu'
    dtype: any = jnp.float16
    use_stable_embedding: bool = False
    no_scale_embedding: bool = True
    decoder_learned_pos: bool = True
    decoder_normalize_before: bool = True
    share_decoder_input_output_embed: bool = True
    # Added
    version: int = 1
    vocab_size: int = 50272
    layer_norm_eps: float = 0.00001
    num_pp_stages: int = None
    # parallelize
    mark_boundary: bool = True


class OPTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert not self.config.use_stable_embedding
        self.embed_scale = 1.0 if self.config.no_scale_embedding else math.sqrt(
            self.config.hidden_size)
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.input_dim,
            dtype=self.dtype,
        )
        assert self.config.max_seq_len is not None
        assert self.config.decoder_learned_pos
        self.position_embeddings = nn.Embed(
            self.config.max_seq_len + self.config.pad + 1,
            self.config.hidden_size,
            dtype=self.dtype,
        )
        self.project_in_dim = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
        ) if self.config.input_dim != self.config.hidden_size else None

    def __call__(self, input_ids, position_ids):
        # Embed
        inputs_embeds = self.embed_scale * self.word_embeddings(
            input_ids.astype("i4"))
        if self.project_in_dim is not None:
            inputs_embeds = self.project_in_dim(inputs_embeds)
        position_embeds = self.position_embeddings(position_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + position_embeds
        return hidden_states


class OPTSelfAttention(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.n_head != 0:
            raise ValueError(
                f"`hidden_size`: {self.config.hidden_size} has to be a "
                f"multiple of `n_head`: {self.config.n_head}"
            )

        self.qkv_combined = nn.Dense(
            self.config.hidden_size * 3,
            dtype=self.dtype,
            use_bias=False,
        )

        # The fused_mmha kernel fuses the bias add, so we do not load the bias in Dense and
        # instead feed it into the kernel.
        head_dim = self.config.hidden_size // self.config.n_head
        self.qkv_combined_bias = self.param(
            'qkv_combined_bias', flax.linen.initializers.zeros,
            (3, self.config.n_head, head_dim), self.dtype)

    def __call__(self,
                 hidden_states,
                 output_attentions: bool = False,
                 attention_cache=None):
        head_dim = self.config.hidden_size // self.config.n_head
        assert attention_cache is not None, "Attention cache must be provided for now"

        # Shape: [1D seq, heads, head_dim, 3]
        qkv_combined_states = self.qkv_combined(hidden_states)
        qkv_combined_states = qkv_combined_states.reshape(
            qkv_combined_states.shape[:1] +
            (self.config.n_head, head_dim, 3))

        # Shape: [1D seq, 3, heads, head_dim]
        qkv_combined_states = qkv_combined_states.transpose((0, 3, 1, 2))

        qkv_combined_states_w_bias = qkv_combined_states + self.qkv_combined_bias

        # Shape of cache_key and cache_value: [batch * max_length, heads, head_dim]
        # Shape of cache_index: [batch * max_length]
        cache_key, cache_value = attention_cache

        # perform_attention = True
        # if perform_attention:
        attn_output = fused_mmha(qkv_combined_states, self.qkv_combined_bias,
                                 cache_key, cache_value)
        # else:
        #     attn_output = jnp.ones((qkv_combined_states.shape[0], qkv_combined_states.shape[2], qkv_combined_states.shape[3]))

        attn_output = attn_output.reshape(attn_output.shape[:1] + (-1,))

        # Update cache key and value. Note that the cache index should
        # be updated outside the model.
        _, key_states, value_states = jnp.split(qkv_combined_states_w_bias,
                                                3,
                                                axis=1)
        attention_cache = (key_states, value_states)

        if output_attentions:
            print("Do not support output_attentions")
        outputs = (attn_output, attention_cache)
        return outputs


class OPTAttention(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        assert self.config.decoder_normalize_before
        self.self = OPTSelfAttention(self.config, dtype=self.dtype)
        self.dense = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 output_attentions: bool = False,
                 attention_cache=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.self(hidden_states,
                                 output_attentions=output_attentions,
                                 attention_cache=attention_cache)
        attn_output = attn_outputs[0]
        attention_cache = attn_outputs[1]
        hidden_states = self.dense(attn_output)
        hidden_states = hidden_states + residual
        outputs = (hidden_states, attention_cache)

        if output_attentions:
            outputs += (attn_outputs[2],)

        return outputs


class OPTFFN(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.fc1 = nn.Dense(
            self.config.ffn_embed_dim,
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.activation_fn]
        self.fc2 = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class OPTTransformerLayer(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert self.config.decoder_normalize_before
        assert not getattr(self.config, "cross_self_attention", False)
        assert not getattr(self.config, "scale_heads", False)
        assert not getattr(self.config, "scale_attn", False)
        assert not getattr(self.config, "scale_fc", False)
        self.attention = OPTAttention(self.config, dtype=self.dtype)
        self.ffn = OPTFFN(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 output_attentions: bool = False,
                 attention_cache=None):

        attention_outputs = self.attention(hidden_states,
                                           output_attentions=output_attentions,
                                           attention_cache=attention_cache)
        attention_output = attention_outputs[0]
        attention_cache = attention_outputs[1]

        hidden_states = self.ffn(attention_output)

        outputs = (hidden_states, attention_cache)

        if output_attentions:
            outputs += (attention_outputs[2],)
        return outputs


class OPTTransformerLayerCollection(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.layers = [
            OPTTransformerLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        new_attention_cache = () if attention_cache is not None else None

        if self.config.num_pp_stages is not None:
            assert self.config.num_hidden_layers % self.config.num_pp_stages == 0
            layers_per_stage = self.config.num_hidden_layers // self.config.num_pp_stages

        for i, layer in enumerate(self.layers):
            if self.config.num_pp_stages is not None:
                if i % layers_per_stage == 0 and i != 0:
                    stage_id = i // layers_per_stage
                    if self.config.mark_boundary:
                        mark_pipeline_boundary()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_attention_cache = None
            if attention_cache is not None:
                layer_attention_cache = attention_cache[i]
            layer_outputs = layer(hidden_states,
                                  output_attentions=output_attentions,
                                  attention_cache=layer_attention_cache)
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

        return OPTModelOutput(last_hidden_state=hidden_states,
                              hidden_states=all_hidden_states,
                              attentions=all_attentions,
                              attention_cache=new_attention_cache)


class OPTTransformerModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert self.config.decoder_normalize_before
        self.embeddings = OPTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = OPTTransformerLayerCollection(self.config,
                                                     dtype=self.dtype)
        if self.config.version > 2:
            self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                           dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
    ):
        hidden_states = self.embeddings(input_ids, position_ids)
        outputs = self.encoder(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_cache=attention_cache,
        )
        hidden_states = outputs[0]
        if self.config.version > 2:
            hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            # if pooled is None, don't return it
            return (hidden_states,) + outputs[1:]

        return OPTModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )


class OPTForLMModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float16
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = OPTTransformerModule(config=self.config,
                                                 dtype=self.dtype)

        self.project_out_dim = nn.Dense(
            self.config.input_dim,
            dtype=self.dtype,
        ) if self.config.input_dim != self.config.hidden_size else None

        if self.config.share_decoder_input_output_embed:
            self.decoder = None
        else:
            self.decoder = nn.Dense(self.config.vocab_size,
                                    dtype=self.dtype,
                                    use_bias=False)

    def __call__(
        self,
        input_ids,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
    ):
        # Model
        outputs = self.transformers(
            input_ids,
            position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_cache=attention_cache,
        )

        hidden_states = outputs[0]

        if self.project_out_dim is not None:
            hidden_states = self.project_out_dim(hidden_states)

        if self.config.share_decoder_input_output_embed:
            if self.dtype == jnp.float16:
                shared_embedding = self.transformers.embeddings.word_embeddings.embedding_fp16
            else:
                shared_embedding = self.transformers.variables["params"][
                    "embeddings"]["word_embeddings"]["embedding"]
            assert self.decoder is None
            logits = hidden_states @ shared_embedding.T
        else:
            assert self.decoder is not None
            logits = self.decoder(hidden_states)

        # Compute the prediction scores
        if not return_dict:
            return (logits,) + outputs[1:]

        return OPTLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )


def init_model_aval(config, total_input_len, total_cache_len):
    """In 1D: we specify total_input_len and total_cache_len in advance."""
    model = OPTForLMModule(config, dtype=config.dtype)
    rngkey = jax.core.ShapedArray((2,), jnp.uint32)
    input_ids = jax.core.ShapedArray((total_input_len,), jnp.int32)
    position_ids = jax.core.ShapedArray((total_input_len,), jnp.int32)
    cache = init_cache_aval(config, total_cache_len)

    params = jax.eval_shape(model.init,
                            rngkey,
                            input_ids,
                            position_ids,
                            attention_cache=cache)
    params = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, config.dtype),
                          params)
    return model, params


def init_cache_aval(config, total_cache_len):
    dtype = config.dtype
    head_dim = config.hidden_size // config.n_head

    all_cache = []
    for i in range(config.num_hidden_layers):
        layer_cache = (
            jax.core.ShapedArray((total_cache_len,
                                  config.n_head, head_dim),
                                 dtype),
            jax.core.ShapedArray((total_cache_len,
                                  config.n_head, head_dim),
                                 dtype),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def init_cache_np(config, total_cache_len):
    """Init cache per sequence with numpy arrays."""
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
    head_dim = config.hidden_size // config.n_head

    all_cache = []
    for i in range(config.num_hidden_layers):
        layer_cache = (
            np.zeros((total_cache_len,
                      config.n_head, head_dim),
                     dtype=np_dtype),
            np.zeros((total_cache_len,
                      config.n_head, head_dim),
                     dtype=np_dtype),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def build_position_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).astype(np.int32)
    position_ids = np.cumsum(mask).astype(np.int32) * mask + padding_idx
    return position_ids


class BatchLevelInputPool:
    """This pool is for batch-level scheduling."""
    def __init__(self,
                 config,
                 batch_size=512,
                 cache_size=512,
                 max_cache_per_seq=128):
        # Opt model config
        self.model_config = config
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.max_cache_per_seq = max_cache_per_seq

        # other useful configs
        self.pad = self.model_config.pad if "pad" in dir(self.model_config) else 1
        self.hidden_dim = self.model_config.hidden_size
        self.vocab_size = self.model_config.vocab_size

        # caches, which is statically allocated
        self.kv_caches = jax.tree_map(jnp.array, init_cache_np(config, self.cache_size))
        self.kv_cache_ids = np.zeros((self.cache_size, ), dtype=np.int32)
        self.kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in self.kv_caches]

        # internal states
        # num_prev_tokens tracks the progress of each sentence.
        self.input_sequence_ids = []
        self.num_prev_tokens = {}
        self.next_cache_index = {}

        # cuda kernels
        self.custom_memcpy = cupyx.jit.rawkernel()(custom_memcpy)
        self.custom_reshape_logits = cupyx.jit.rawkernel()(custom_reshape_logits)

    def enter_prompts(self, input_sequences: List[List[int]]):
        # reset cache, sentence_ids, etc.
        self.reset()
        # check input has no padding
        for seq in input_sequences:
            assert self.pad not in seq
        # generate IDs for them
        self.input_sequence_ids = [i + 1 for i in range(0, len(input_sequences))]

        # update num_prev_tokens for the sentence
        # for i, sentence_id in enumerate(sentence_ids):
        for i, id in enumerate(self.input_sequence_ids):
            assert id not in self.num_prev_tokens
            self.num_prev_tokens[id] = 0
            assert id not in self.next_cache_index
            self.next_cache_index[id] = i * self.max_cache_per_seq + self.num_prev_tokens[id]
        input, input_index, position_ids = self._generate_1d_inputs(input_sequences)
        return input, input_index, position_ids

    def enter_decoding(self, input_sequences: List[List[int]]):
        # do some check:
        assert len(input_sequences) == len(self.input_sequence_ids)
        for i, id in enumerate(self.input_sequence_ids):
            assert id in self.num_prev_tokens
            assert id in self.next_cache_index
            assert len(input_sequences[i]) == 1
            assert self.num_prev_tokens[id] > 0
        input, input_index, position_ids = self._generate_1d_inputs(input_sequences)
        return input, input_index, position_ids

    def reshape_logits_legacy(self, logits, input_index, padded_shape):
        """Unflatten the 1D logits output to be 2D."""
        ret_shape = padded_shape + (logits.shape[-1],)
        logits_cupy = jax_tensor_to_cupy(logits)
        ret_cupy = cupy.zeros(ret_shape, dtype=logits_cupy.dtype)
        index = 0
        for i in range(1, padded_shape[0] + 1):
            num_elements = np.sum(input_index==i)
            ret_cupy[i-1, :num_elements, :] = logits_cupy[index:index+num_elements, :]
            ret_cupy[i-1, num_elements:, :] = logits_cupy[index+num_elements-1,:]
            index = index + num_elements
        ret = cupy.asnumpy(ret_cupy)
        return ret

    def reshape_logits(self, logits, unpadded_input, padded_shape):
        """Reshape the 1D logits output to be 2D."""
        src_logits = jax_tensor_to_cupy(logits)
        dst_shape = padded_shape + (self.vocab_size, )
        num_blocks = dst_shape[0] * dst_shape[1]
        seq_lens = [len(seq) for seq in unpadded_input]
        src_indices = []
        start = 0
        for seq_index, seq_len in enumerate(seq_lens):
            end = start + seq_len
            src_indices.extend(list(range(start, end)) + [end - 1] * (dst_shape[1] - seq_len))
            start = end
        src_indices = cupy.array(src_indices)
        dst_logits = cupy.zeros(dst_shape, dtype=logits.dtype)
        self.custom_reshape_logits[num_blocks, 1024](dst_logits.ravel(), src_logits.ravel(), src_indices)
        ret = cupy.asnumpy(dst_logits)
        return ret

    def _generate_1d_inputs(self, input_sequences: List[List[int]]):
        """Generate the three elements: input tokens, input token index, and position_ids"""
        input = sum(input_sequences, [])
        assert len(input) <= self.batch_size, "Please allocate a larger batch size"
        input_list = input + [self.pad] * (self.batch_size - len(input))
        input = np.array(input_list, dtype=np.int32)

        # generate an index array that tells the sentence id of each token
        assert len(input_sequences) == len(self.input_sequence_ids)
        input_index = []
        for i, sentence_id in enumerate(self.input_sequence_ids):
            input_index.extend([sentence_id] * len(input_sequences[i]))
        assert len(input_index) <= self.batch_size
        input_index = np.array(input_index + [0] * (self.batch_size - len(input_index)), dtype=np.int32)

        # generate position ids
        position_ids = []
        for i, sentence_id in enumerate(self.input_sequence_ids):
            start_idx = 1 + self.pad + self.num_prev_tokens[sentence_id]
            position_ids.extend(list(range(start_idx, start_idx + len(input_sequences[i]))))
        position_ids = position_ids + [self.pad] * (self.batch_size - len(position_ids))
        position_ids = np.array(position_ids, dtype=jnp.int32)
        return input, input_index, position_ids

    def reset(self):
        self.num_prev_tokens = {}
        self.next_cache_index = {}
        self.input_sequence_ids = []
        for k, v in self.kv_caches_cupy:
            k.fill(0.0)
            v.fill(0.0)
        self.kv_cache_ids.fill(0)

    def update_cache(self, input_sequences: List[List[int]], kv):
        num_threads_per_block = 256
        dst_indices = []
        for i, sentence_id in enumerate(self.input_sequence_ids):
            start = self.next_cache_index[sentence_id]
            dst_indices.extend(list(range(start, start + len(input_sequences[i]))))
        dst_indices = cupy.array(dst_indices)
        sum_src_len = sum(len(seq) for seq in input_sequences)
        # Note(Hao): this jax -> cupy conversion has a little overhead, though
        src_kv = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv]
        for layer_idx, (src_k, src_v) in enumerate(src_kv):
            dst_k, dst_v = self.kv_caches_cupy[layer_idx]
            self.custom_memcpy[sum_src_len, num_threads_per_block](dst_k.ravel(), dst_v.ravel(),
                                                                   src_k.ravel(), src_v.ravel(),
                                                                   dst_indices, self.hidden_dim)
        for i, sentence_id in enumerate(self.input_sequence_ids):
            start = self.next_cache_index[sentence_id]
            self.kv_cache_ids[start:start+len(input_sequences[i])] = sentence_id

    def update_num_prev_tokens(self, input_sequences: List[List[int]]):
        for i, sentence_id in enumerate(self.input_sequence_ids):
            self.num_prev_tokens[sentence_id] += len(input_sequences[i])
        for i, sentence_id in enumerate(self.input_sequence_ids):
            self.next_cache_index[sentence_id]  = (sentence_id - 1) * \
                                                  self.max_cache_per_seq + self.num_prev_tokens[sentence_id]


class PromptStatus(Enum):
    PROMPT = 1
    DECODING = 2
    FINISHED = 3


class Prompt:
    def __init__(self, input_ids, sentence_id):
        self.input_ids = input_ids
        self.sentence_id = sentence_id
        self.status = PromptStatus.PROMPT
        self.cache_start_index = None
        # states to be filled during generation
        self.generated_ids = []
        self.last_generated_id = None

        # latency information
        self.start_time = None
        self.finish_time = None

    def finish(self, finish_token_id):
        self.finish_time = time.time()
        self.status = PromptStatus.FINISHED
        self.generated_ids.append(finish_token_id)
        self.last_generated_id = finish_token_id

    def init_cache(self, cache_index):
        self.cache_start_index = cache_index

    def add_token(self, token_id):
        if self.status == PromptStatus.PROMPT:
            self.status = PromptStatus.DECODING
        else:
            assert self.last_generated_id is not None and self.status == PromptStatus.DECODING
            self.generated_ids.append(self.last_generated_id)
        self.last_generated_id = token_id

    def start(self):
        self.start_time = time.time()

    @property
    def prompt_length(self):
        return len(self.input_ids)

    @property
    def generation_length(self):
        return len(self.generated_ids)

    @property
    def num_prev_tokens(self):
        if self.status == PromptStatus.PROMPT:
            return 0
        else:
            return self.prompt_length + self.generation_length

    @property
    def cache_end_index(self):
        if self.cache_start_index == None:
            raise RuntimeError("The prompt has not been assigned a cache slot.")
        return self.cache_start_index + self.generation_length + self.prompt_length

    @property
    def latency(self):
        if self.status != PromptStatus.FINISHED:
            raise RuntimeError("Unfinished prompt.")
        return self.finish_time - self.start_time

    def print(self):
        print(self.input_ids + ":" + self.generated_ids)


class Cache:
    def __init__(self, cache_size, max_cache_per_seq, model_config):
        self.cache_size = cache_size
        self.max_cache_per_seq = max_cache_per_seq
        self.model_config = model_config

        # init the cache
        self.kv_caches = jax.tree_map(jnp.array, init_cache_np(model_config, self.cache_size))
        self.kv_cache_ids = np.zeros((self.cache_size, ), dtype=np.int32)
        self.kv_caches_cupy = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in self.kv_caches]

        self.num_slot = self.cache_size // self.max_cache_per_seq
        # Stores the starting indices of all vacant cache slots
        # A slot can be indexed as [start_index, start_index + self.max_cache_per_seq]
        self.vacancies: List[int] = list(range(0, self.num_slot * self.max_cache_per_seq,
                                         self.max_cache_per_seq))
        heapq.heapify(self.vacancies)

        # Track the index of the first
        self.continuity_tracker = 0
        self.hidden_dim = self.model_config.hidden_size

        # kernels.
        self.custom_memcpy_v2 = cupyx.jit.rawkernel()(custom_memcpy_v2)
        self.custom_mv = cupyx.jit.rawkernel()(custom_mv)

    def release_slot(self, slot: int):
        """Release a cache slot given its starting index."""
        assert slot not in self.vacancies
        heapq.heappush(self.vacancies, slot)
        # reset the cache indices of this slot as there
        self.kv_cache_ids[slot:slot + self.max_cache_per_seq] = 0
        if self.continuity_tracker - slot == self.max_cache_per_seq:
            # meaning the released slot is next to the tracker
            new_start = slot - self.max_cache_per_seq
            while new_start in self.vacancies:
                new_start = new_start - self.max_cache_per_seq
            self.continuity_tracker = new_start + self.max_cache_per_seq

    def take_slot(self):
        """Allocate the first available slot and return the slot index."""
        if len(self.vacancies) == 0:
            raise RuntimeError("There is no vacancy in cache.")
        slot = heapq.heappop(self.vacancies)
        if slot == self.continuity_tracker:
            self.continuity_tracker = slot + self.max_cache_per_seq
        return slot

    def erase(self):
        self.vacancies = list(range(0, self.num_slot * self.max_cache_per_seq, self.max_cache_per_seq))
        heapq.heapify(self.vacancies)
        self.continuity_tracker = 0
        self.kv_cache_ids.fill(0)

    def is_continuous(self):
        """If all the vacant slots in cache are continous."""
        if len(self.vacancies) == 0:
            return True
        heap_top = self.vacancies[0]
        return heap_top == self.continuity_tracker

    def get_continuation_plan(self):
        """Find a movement plan to continuzie the cache.

        Returns:
            dst_slots (OrderedSet): the starting indices of the destination slots
            src_slots (OrderedSet): the starting indices of the source slots
        """
        # find all vacancies that are before continuity_tracker
        vacancies_to_fill = OrderedSet(v for v in self.vacancies if v < self.continuity_tracker)
        start = self.continuity_tracker - len(vacancies_to_fill) * self.max_cache_per_seq
        proposals = OrderedSet(range(start, self.continuity_tracker, self.max_cache_per_seq))
        dst_slots = OrderedSet(v for v in vacancies_to_fill if v not in proposals)
        fake_vacancies = vacancies_to_fill.difference(dst_slots)
        src_slots = proposals.difference(fake_vacancies)
        assert len(dst_slots) == len(src_slots)
        return dst_slots, src_slots

    def update_cache(self, kv, src_indices, dst_indices, src_sentence_ids):
        num_threads_per_block = 256
        dst_indices_cupy = cupy.array(dst_indices)
        src_indices_cupy = cupy.array(src_indices)
        src_kv = [(jax_tensor_to_cupy(k), jax_tensor_to_cupy(v)) for k, v in kv]
        for layer_idx, (src_k, src_v) in enumerate(src_kv):
            dst_k, dst_v = self.kv_caches_cupy[layer_idx]
            self.custom_memcpy_v2[len(src_indices), num_threads_per_block](
                dst_k.ravel(), dst_v.ravel(), src_k.ravel(), src_v.ravel(),
                dst_indices_cupy, src_indices_cupy, self.hidden_dim)
        # update cache ids
        self.kv_cache_ids[dst_indices] = src_sentence_ids

    def continuize(self, dst_slots, src_slots):
        """Move cache contents from src to dst."""
        src_indices = sum([list(range(src_slot, src_slot + self.max_cache_per_seq)) for src_slot in src_slots], [])
        dst_indices = sum([list(range(dst_slot, dst_slot + self.max_cache_per_seq)) for dst_slot in dst_slots], [])
        self.kv_cache_ids[dst_indices] = self.kv_cache_ids[src_indices]
        self.kv_cache_ids[src_indices] = 0

        num_threads_per_block = 256
        src_indices_cupy = cupy.array(src_indices)
        dst_indices_cupy = cupy.array(dst_indices)

        for layer_idx, (k, v) in enumerate(self.kv_caches_cupy):
            self.custom_mv[len(src_indices), num_threads_per_block](
                k.ravel(), v.ravel(), dst_indices_cupy, src_indices_cupy,
                self.hidden_dim)

        # update vacancies
        for slot in src_slots:
            if slot not in self.vacancies:
                self.release_slot(slot)
        for slot in dst_slots:
            if slot in self.vacancies:
                self.vacancies.remove(slot)
        # update continuity tracker
        self.continuity_tracker = self.vacancies[0]


class IterationLevelInputPool:
    """This pool is for iteration-level scheduling."""
    def __init__(self,
                 input_pool_config,
                 model_config,
                 max_length=None,
                 max_new_tokens=None):
        self.batch_size = input_pool_config.batch_size
        self.cache_size = input_pool_config.cache_size
        self.max_cache_per_seq = input_pool_config.max_cache_per_seq
        self.model_config = model_config
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        if self.max_length and self.max_length > self.max_cache_per_seq:
            warnings.warn("`max_length` is greater than `max_cache_per_seq`.")
        if self.max_new_tokens and self.max_new_tokens > self.max_cache_per_seq:
            warnings.warn("`max_new_tokens` is greater than `max_cache_per_seq`.")

        self.cache = Cache(self.cache_size, self.max_cache_per_seq, self.model_config)

        # input pool states
        self.todo = queue.Queue()
        self.wip = OrderedSet()
        self.done = OrderedSet()

        # current batch state
        self._current_batch = None
        self._sentence_id_counter = 1

        # model config
        self.pad = self.model_config.pad if "pad" in dir(self.model_config) else 1
        self.eos = self.model_config.eos_token_id if "eos_token_id" in dir(self.model_config) else 2

    def is_finished(self):
        return self.todo.empty() and len(self.wip) == 0

    def enter_prompts(self, input_sequences: List[List[int]]):
        """Enter a new batch of prompts into self.todo."""
        sentence_ids = self.next_sentence_id(len(input_sequences))
        for i, seq in enumerate(input_sequences):
            p = Prompt(seq, sentence_ids[i])
            self.todo.put(p)

    def next(self):
        decoding_input = []
        # figure out WIP prompts and put their next token in a list
        for p in self.wip:
            assert p.status == PromptStatus.DECODING, \
                "WIP queue must have all prompts in decoding status."
            decoding_input.append(p)

        # pop out new prompts, concat them into a list
        prompt_input = []
        for i in range(self.num_vacant_sequence_slot):
            if self.todo.empty():
                break
            p = self.todo.get()
            assert p.status == PromptStatus.PROMPT, \
                "unfinished queue must have all prompts in PROMPT status. "
            prompt_input.append(p)

        # make input: prompts must go first
        input = sum([p.input_ids for p in prompt_input], []) + [p.last_generated_id for p in decoding_input]
        input = np.array(input + [self.pad] * (self.batch_size - len(input)), dtype=np.int32)

        # make input index
        input_index = []
        for p in prompt_input:
            input_index.extend([p.sentence_id] * p.prompt_length)
        for p in decoding_input:
            input_index.append(p.sentence_id)
        input_index = np.array(input_index + [0] * (self.batch_size - len(input_index)), dtype=np.int32)

        # make position ids
        position_ids = []
        for p in prompt_input:
            start_idx = 1 + self.pad + p.num_prev_tokens
            position_ids.extend(list(range(start_idx, start_idx + p.prompt_length)))
        for p in decoding_input:
            start_idx = 1 + self.pad + p.num_prev_tokens
            position_ids.extend(list(range(start_idx, start_idx + 1)))
        position_ids = np.array(position_ids +  [0] * (self.batch_size - len(position_ids)), dtype=np.int32)

        # start prompts
        for p in prompt_input:
            p.start()

        self._current_batch = prompt_input + decoding_input
        logger.debug(f"This batch has {len(prompt_input)} new prompts {[p.sentence_id for p in prompt_input]}. "
                     f"and {len(decoding_input)} in decoding {[p.sentence_id for p in decoding_input]}. "
                     f"Todo size: {self.todo.qsize()}, cache vacancies: {[v // self.cache.max_cache_per_seq for v in self.cache.vacancies]}, "
                     f"continuity from: {self.cache.continuity_tracker // self.cache.max_cache_per_seq}")

        logit_positions = []
        i = -1
        for p in prompt_input:
            i += p.prompt_length
            logit_positions.append(i)
        for _ in decoding_input:
            i += 1
            logit_positions.append(i)

        # return inputs
        return input, input_index, position_ids, logit_positions

    def update_cache(self, kv, generated_ids):
        if self._current_batch is None:
            raise RuntimeError("There is no pending batch so update_cache should not be called.")

        # we need to copy cache using one custom kernel, so we record the src and dst indices
        src_indices = []
        dst_indices = []
        src_sentence_ids = []

        # check EOS, move finished sentences from wip to finished queue
        read_idx = 0
        for generated_id, p in zip(generated_ids, self._current_batch):
            if self.check_exit_condition(p, generated_id):
                if p.status == PromptStatus.DECODING:
                    assert p in self.wip
                    self.wip.remove(p)
                    read_idx += 1
                if p.status == PromptStatus.PROMPT:
                    read_idx += p.prompt_length
                exit_reason = "EOS" if generated_id == self.eos else "reaching max length"
                logger.debug(f"Prompt {p.sentence_id} exits because of {exit_reason}. "
                             f"Release cache {p.cache_start_index // self.max_cache_per_seq}" )
                p.finish(generated_id)
                self.done.add(p)
                # For finished sentences, release cache slot, update cache index
                self.cache.release_slot(p.cache_start_index)
            elif p.status == PromptStatus.PROMPT:
                # PROMPT -> DECODING
                p.init_cache(self.cache.take_slot())
                dst_indices.extend(list(range(p.cache_start_index, p.cache_end_index)))
                src_indices.extend(list(range(read_idx, read_idx + p.prompt_length)))
                assert (p.cache_end_index - p.cache_start_index) == p.prompt_length
                src_sentence_ids.extend([p.sentence_id] * p.prompt_length)
                p.add_token(generated_id)
                self.wip.add(p)
                read_idx += p.prompt_length
            elif p.status == PromptStatus.DECODING:
                # DECODING -> DECODING
                src_indices.append(read_idx)
                dst_indices.append(p.cache_end_index)
                src_sentence_ids.append(p.sentence_id)
                p.add_token(generated_id)
                read_idx += 1
            else:
                raise RuntimeError(f"Prompt status: {p.status} should not appear here." )

        # no cache writing task
        if len(src_indices) == 0:
            logger.debug("All prompts have finished. Cache will be erased.")
            self.cache.erase()
            return

        # update cache
        self.cache.update_cache(kv, src_indices, dst_indices, src_sentence_ids)

        reorg_dst_slots, reorg_src_slots = self.cache.get_continuation_plan()
        if len(reorg_dst_slots) > 0:
            self.cache.continuize(reorg_dst_slots, reorg_src_slots)
            # update the prompt that has been influenced
            for dst_slot, src_slot in zip(reorg_dst_slots, reorg_src_slots):
                for p in self.wip:
                    if p.cache_start_index == src_slot:
                        p.cache_start_index = dst_slot
                        break
            # Note(Hao): the cache order must align with input order.
            # so we have to sort self._current_batch based on the order of cache
            self.wip = OrderedSet(sorted(self.wip, key=lambda x: x.cache_start_index, reverse=False))

    def get_results(self):
        """Return results sorted by their sentence id."""
        sorted_results = sorted(self.done, key=lambda x: x.sentence_id, reverse=False)
        return [p.input_ids + p.generated_ids for p in sorted_results]

    def next_sentence_id(self, number):
        counter = self._sentence_id_counter
        if number == 1:
            ret = [counter]
        else:
            ret = list(range(counter, counter + number))
        self._sentence_id_counter = (counter + number) % (1 << 60)
        return ret

    @property
    def num_vacant_sequence_slot(self):
        """Return the global vacancy."""
        return min(self.batch_size - len(self.wip), len(self.cache.vacancies))

    def check_exit_condition(self, prompt, generated_id):
        """Check Exit condition: reaching EOS or reaching max length."""
        if generated_id == self.eos:
            return True
        if self.max_new_tokens:
            if prompt.generation_length + 1 == self.max_new_tokens:
                return True
        if self.max_length:
            if prompt.generation_length + 1 + prompt.prompt_length == self.max_length:
                return True
        return False


def custom_memcpy(dst_k, dst_v, src_k, src_v, dst_indices, hidden_dim):
    thread_idx = cupyx.jit.threadIdx.x
    src_idx = cupyx.jit.blockIdx.x
    dst_idx = dst_indices[src_idx]
    num_elements_per_thread = (hidden_dim + 256 - 1) // 256
    for i in range(num_elements_per_thread):
        j = thread_idx + 256 * i
        if j < hidden_dim:
            dst_k[dst_idx * hidden_dim + j] = src_k[src_idx * hidden_dim + j]
            dst_v[dst_idx * hidden_dim + j] = src_v[src_idx * hidden_dim + j]


def custom_memcpy_v2(dst_k, dst_v, src_k, src_v, dst_indices, src_indices, hidden_dim):
    thread_idx = cupyx.jit.threadIdx.x
    src_idx = src_indices[cupyx.jit.blockIdx.x]
    # src_idx = cupyx.jit.blockIdx.x
    dst_idx = dst_indices[cupyx.jit.blockIdx.x]
    num_elements_per_thread = (hidden_dim + 256 - 1) // 256
    for i in range(num_elements_per_thread):
        j = thread_idx + 256 * i
        if j < hidden_dim:
            dst_k[dst_idx * hidden_dim + j] = src_k[src_idx * hidden_dim + j]
            dst_v[dst_idx * hidden_dim + j] = src_v[src_idx * hidden_dim + j]


def custom_mv(k, v, dst_indices, src_indices, hidden_dim):
    """This custom kernel moves contents at src_indices to dst_indices."""
    thread_idx = cupyx.jit.threadIdx.x
    block_idx = cupyx.jit.blockIdx.x
    src_idx = src_indices[block_idx]
    dst_idx = dst_indices[block_idx]
    num_elements_per_thread = (hidden_dim + 256 - 1) // 256
    for i in range(num_elements_per_thread):
        j = thread_idx + 256 * i
        if j < hidden_dim:
            k[dst_idx * hidden_dim + j] = k[src_idx * hidden_dim + j]
            k[src_idx * hidden_dim + j] = 0.0
            v[dst_idx * hidden_dim + j] = v[src_idx * hidden_dim + j]
            v[src_idx * hidden_dim + j] = 0.0


def custom_reshape_logits(dst, src, indices):
    thread_idx = cupyx.jit.threadIdx.x
    dst_idx = cupyx.jit.blockIdx.x
    src_idx = indices[dst_idx]
    vocab_size = 50272
    num_elements_per_thread = (vocab_size + 1024 - 1) // 1024
    for i in range(num_elements_per_thread):
        j = thread_idx + 1024 * i
        if j < vocab_size:
            dst[dst_idx * vocab_size + j] = src[src_idx * vocab_size + j]


def unpad(inputs: Union[np.ndarray, torch.Tensor, List[List[int]]], pad=1):
    if isinstance(inputs, np.ndarray) or isinstance(inputs, torch.Tensor):
        inputs = inputs.tolist()
    unpadded_inputs = []
    for seq in inputs:
        if pad in seq:
            unpadded_inputs.append(seq[:seq.index(pad)])
        else:
            unpadded_inputs.append(seq)
    return unpadded_inputs


def pad(inputs: Union[np.ndarray, torch.Tensor, List[List[int]]], pad=1):
    if isinstance(inputs, np.ndarray) or isinstance(inputs, torch.Tensor):
        inputs = inputs.tolist()
    padded_inputs = []
    target_len = max(len(seq) for seq in inputs)
    for seq in inputs:
        if len(seq) < target_len:
            padded_inputs.append(seq + [pad] * (target_len - len(seq)))
        else:
            padded_inputs.append(seq)
    return padded_inputs


def load_params_np(params, path, config, dummy=False):
    """Load parameterswith numpy arrays."""
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
    if dummy:
        return jax.tree_map(lambda x: np.full(x.shape, 1e-9, np_dtype), params)

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array):
        param_dict = params
        param_keys = param_key.split('.')
        for i, key in enumerate(param_keys):
            if i == len(param_keys) - 1:
                if dummy:
                    param_dict[key] = jax.core.ShapedArray(
                        param_dict[key].shape, param_dict[key].dtype)
                else:
                    assert param_dict[key].shape == loaded_array.shape
                    #assert param_dict[key].dtype == loaded_array.dtype
                    param_dict[key] = loaded_array
            else:
                param_dict = param_dict[key]

    head = config.n_head
    head_dim = config.hidden_size // head

    params = params.unfreeze()
    load_param("params.transformers.embeddings.word_embeddings.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("params.transformers.embeddings.position_embeddings.embedding",
               load_array("decoder.embed_positions.weight"))
    if config.version > 2:
        load_param("params.transformers.layer_norm.scale",
                   load_array("decoder.layer_norm.weight"))
        load_param("params.transformers.layer_norm.bias",
                   load_array("decoder.layer_norm.bias"))
    for i in range(config.num_hidden_layers):
        param_prefix = f"params.transformers.encoder.{i}."
        load_prefix = f"decoder.layers.{i}."
        # Attention weights
        wq = load_array(load_prefix + "self_attn.q_proj.weight")
        wk = load_array(load_prefix + "self_attn.k_proj.weight")
        wv = load_array(load_prefix + "self_attn.v_proj.weight")
        dim = wq.shape[-1]
        w_qkv = np.concatenate([wq, wk, wv], axis=0).reshape(
            (3, -1, dim)).transpose([2, 1, 0]).reshape((dim, -1))
        load_param(param_prefix + "attention.self.qkv_combined.kernel", w_qkv)
        bq = load_array(load_prefix + "self_attn.q_proj.bias")
        bk = load_array(load_prefix + "self_attn.k_proj.bias")
        bv = load_array(load_prefix + "self_attn.v_proj.bias")
        # b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
        #     (3, dim)).transpose([1, 0]).reshape((-1,))
        # load_param(param_prefix + "attention.self.qkv_combined.bias", b_qkv)
        b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
            (3, head, head_dim)).astype(np_dtype)
        load_param(param_prefix + "attention.self.qkv_combined_bias", b_qkv)
        load_param(
            param_prefix + "attention.dense.kernel",
            np.transpose(load_array(load_prefix + "self_attn.out_proj.weight")))
        load_param(param_prefix + "attention.dense.bias",
                   load_array(load_prefix + "self_attn.out_proj.bias"))
        load_param(param_prefix + "attention.layer_norm.scale",
                   load_array(load_prefix + "self_attn_layer_norm.weight"))
        load_param(param_prefix + "attention.layer_norm.bias",
                   load_array(load_prefix + "self_attn_layer_norm.bias"))
        # FFN weights
        load_param(param_prefix + "ffn.fc1.bias",
                   load_array(load_prefix + "fc1.bias"))
        load_param(param_prefix + "ffn.fc1.kernel",
                   np.transpose(load_array(load_prefix + "fc1.weight")))
        load_param(param_prefix + "ffn.fc2.bias",
                   load_array(load_prefix + "fc2.bias"))
        load_param(param_prefix + "ffn.fc2.kernel",
                   np.transpose(load_array(load_prefix + "fc2.weight")))
        load_param(param_prefix + "ffn.layer_norm.scale",
                   load_array(load_prefix + "final_layer_norm.weight"))
        load_param(param_prefix + "ffn.layer_norm.bias",
                   load_array(load_prefix + "final_layer_norm.bias"))

    return flax.core.freeze(params)


def get_jax_executable(config: OPTConfig,
                       output_attentions: bool = False,
                       output_hidden_states: bool = False):
    """Get a single-gpu executable."""
    # Note(Hao):
    model, params = init_model_aval(config, total_input_len=256, total_cache_len=512)

    @jax.jit
    def inference_step(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"],
                             # attention_mask=batch["mask"],
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states)
        return output.logits, output.attention_cache

    # executables = {}
    # for length in encoder_chunk_sizes:
    #     executables[length] = inference_step
    return inference_step, params


# def get_pipeshard_executable(config,
#                              batch_size=1,
#                              num_micro_batches=1,
#                              decoding_length_per_step=1024,
#                              support_output_attentions=False,
#                              support_output_hidden_states=False,
#                              autoregressive=True):
#
#     # Init model
#     model, params = init_model_aval(config)
#
#     # Parallelize
#     method = alpa.PipeshardParallel(
#         num_micro_batches=num_micro_batches,
#         pipeline_schedule="inference",
#         layer_option="manual",
#         default_auto_sharding_option=alpa.AutoShardingOption(
#             # Force operator model parallel
#             force_batch_dim_to_mesh_dim=None,
#             # Disabling all-to-all and all-gather generates better intra-op strategies.
#             allow_all_to_all=False,
#             allow_all_gather=False,
#         ))
#
#     if autoregressive:
#
#         @alpa.parallelize(batch_argnums=(1,), method=method)
#         def inference_step_with_cache(params, batch):
#             output = model.apply(
#                 params,
#                 batch["input_ids"],
#                 batch["position_ids"],
#                 attention_cache=batch["cache"],
#                 output_attentions=support_output_attentions,
#                 output_hidden_states=support_output_hidden_states)
#             return output
#
#         alpa.global_config.always_donate_micro_batch_vars = False
#         executable = inference_step_with_cache.get_executable(
#             params, {
#                 "input_ids":
#                     jax.core.ShapedArray((batch_size, 1), jnp.int32),
#                 "position_ids":
#                     jax.core.ShapedArray((batch_size, 1), jnp.int32),
#                 "cache":
#                     init_cache_aval(config, batch_size),
#             })
#     else:
#
#         @alpa.parallelize(batch_argnums=(1,), method=method)
#         def inference_step(params, batch):
#             output = model.apply(
#                 params,
#                 batch["input_ids"],
#                 batch["position_ids"],
#                 output_attentions=support_output_attentions,
#                 output_hidden_states=support_output_hidden_states)
#             return output
#
#         assert batch_size % num_micro_batches == 0, "cannot divide batch_size by num_micro_batches"
#         micro_batch_size = batch_size // num_micro_batches
#
#         executable = inference_step.get_executable(
#             params, {
#                 "input_ids":
#                     jax.core.ShapedArray(
#                         (batch_size, decoding_length_per_step), jnp.int32),
#                 "position_ids":
#                     jax.core.ShapedArray(
#                         (batch_size, decoding_length_per_step), jnp.int32),
#             })
#
#     executable.dump_debug_info("tmp")
#     return executable, params
#
#
# def load_opt_params_worker_func(self, path, prefix_to_idx, config, shapes,
#                                 uuids, indices, mesh_ids):
#     """The worker function to load OPT parameters."""
#
#     def load_array(key):
#         return np.load(os.path.join(path, key))
#
#     def load_param(param_key, loaded_array):
#         i = prefix_to_idx[param_key]
#
#         for j in range(len(mesh_ids[i])):
#             if self.mesh_id != mesh_ids[i][j]:
#                 continue
#
#             assert shapes[i][j] == loaded_array.shape
#             uuid = uuids[i][j]
#             datas = []
#             for k in range(len(self.local_devices)):
#                 idx = self.host_id * len(self.local_devices) + k
#                 datas.append(loaded_array[indices[i][j][idx]])
#             self.put_buffers(uuid, datas)
#
#     load_param("params.transformers.embeddings.word_embeddings.embedding",
#                load_array("decoder.embed_tokens.weight"))
#     load_param("params.transformers.embeddings.position_embeddings.embedding",
#                load_array("decoder.embed_positions.weight"))
#
#     if config.version > 2:
#         load_param("params.transformers.layer_norm.scale",
#                    load_array("decoder.layer_norm.weight"))
#         load_param("params.transformers.layer_norm.bias",
#                    load_array("decoder.layer_norm.bias"))
#
#     layers_per_stage = config.num_hidden_layers // config.num_pp_stages
#     head = config.n_head
#     head_dim = config.hidden_size // head
#
#     for i in range(config.num_hidden_layers):
#         stage_id = i // layers_per_stage
#         if stage_id != self.mesh_id:
#             continue
#
#         param_prefix = f"params.transformers.encoder.{i}."
#         load_prefix = f"decoder.layers.{i}."
#         # Attention weights
#         wq = load_array(load_prefix + "self_attn.q_proj.weight")
#         wk = load_array(load_prefix + "self_attn.k_proj.weight")
#         wv = load_array(load_prefix + "self_attn.v_proj.weight")
#         dim = wq.shape[-1]
#         w_qkv = np.concatenate([wq, wk, wv], axis=0).reshape(
#             (3, -1, dim)).transpose([2, 1, 0]).reshape((dim, -1))
#         load_param(param_prefix + "attention.self.qkv_combined.kernel", w_qkv)
#         bq = load_array(load_prefix + "self_attn.q_proj.bias")
#         bk = load_array(load_prefix + "self_attn.k_proj.bias")
#         bv = load_array(load_prefix + "self_attn.v_proj.bias")
#         # b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
#         #     (3, dim)).transpose([1, 0]).reshape((-1,))
#         b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
#             (3, head, head_dim))
#         load_param(param_prefix + "attention.self.qkv_combined_bias", b_qkv)
#         load_param(
#             param_prefix + "attention.dense.kernel",
#             np.transpose(load_array(load_prefix + "self_attn.out_proj.weight")))
#         load_param(param_prefix + "attention.dense.bias",
#                    load_array(load_prefix + "self_attn.out_proj.bias"))
#         load_param(param_prefix + "attention.layer_norm.scale",
#                    load_array(load_prefix + "self_attn_layer_norm.weight"))
#         load_param(param_prefix + "attention.layer_norm.bias",
#                    load_array(load_prefix + "self_attn_layer_norm.bias"))
#         # FFN weights
#         load_param(param_prefix + "ffn.fc1.bias",
#                    load_array(load_prefix + "fc1.bias"))
#         load_param(param_prefix + "ffn.fc1.kernel",
#                    np.transpose(load_array(load_prefix + "fc1.weight")))
#         load_param(param_prefix + "ffn.fc2.bias",
#                    load_array(load_prefix + "fc2.bias"))
#         load_param(param_prefix + "ffn.fc2.kernel",
#                    np.transpose(load_array(load_prefix + "fc2.weight")))
#         load_param(param_prefix + "ffn.layer_norm.scale",
#                    load_array(load_prefix + "final_layer_norm.weight"))
#         load_param(param_prefix + "ffn.layer_norm.bias",
#                    load_array(load_prefix + "final_layer_norm.bias"))
#
#
# setattr(MeshHostWorker, "load_opt_params_worker_func",
#         load_opt_params_worker_func)
#
#
# def load_params_dis_array(path, executable, params_aval, config, dummy=False):
#     """Load parameters with distributed arrays."""
#     if dummy:
#         alpa.global_config.use_dummy_value_for_benchmarking = True
#         params_info, _ = executable.get_input_placement_specs()
#         flat_args, in_tree = tree_flatten(params_aval)
#         flat_info = tree_leaves(params_info)
#         if hasattr(executable, "mesh_group"):
#             ret = executable.mesh_group.shard_args_to_arrays(
#                 flat_info, flat_args)
#         else:
#             ret = executable.physical_mesh.shard_args_to_arrays_ps(
#                 flat_info, flat_args)
#         alpa.global_config.use_dummy_value_for_benchmarking = False
#         return ret
#
#     params_info, _ = executable.get_input_placement_specs()
#
#     prefix_to_flat_idx = {}
#     ct = itertools.count()
#
#     def dfs(dict_tree, result_dict, cur_prefix):
#         if isinstance(dict_tree, (dict, flax.core.FrozenDict)):
#             for key in dict_tree.keys():
#                 dfs(dict_tree[key], result_dict,
#                     cur_prefix + ("." if cur_prefix else "") + key)
#         else:
#             result_dict[cur_prefix] = next(ct)
#
#     dfs(params_aval, prefix_to_flat_idx, "")
#
#     flat_infos, in_tree = tree_flatten(params_info)
#
#     flat_shapes = []
#     flat_uuids = []
#     flat_indices = []
#     flat_mesh_ids = []
#     flat_arrays = []
#
#     mesh_group = executable.mesh_group
#
#     for info in flat_infos:
#         aval = info.aval
#         if len(info.mesh_ids) == 1:
#             mesh, spec = mesh_group[info.mesh_ids[0]], info.sharding_specs[0]
#             indices = pxla.spec_to_indices(aval.shape, spec)
#             ary_refs, ary_uuid = create_remote_array_refs(mesh)
#             flat_shapes.append([aval.shape])
#             flat_uuids.append([ary_uuid[0]])
#             flat_indices.append([indices])
#             flat_mesh_ids.append([mesh.mesh_id])
#             flat_arrays.append(
#                 DistributedArray(mesh, aval, spec, ary_refs[0], indices))
#         else:
#             tmp_shapes = []
#             tmp_uuids = []
#             tmp_indices = []
#             tmp_mesh_ids = []
#             tmp_arrays = []
#             tmp_meshes = []
#             for mesh_id, spec in zip(info.mesh_ids, info.sharding_specs):
#                 mesh = mesh_group[mesh_id]
#                 indices = pxla.spec_to_indices(aval.shape, spec)
#                 ary_refs, ary_uuid = create_remote_array_refs(mesh)
#                 array = DistributedArray(mesh, aval, spec, ary_refs[0], indices)
#                 tmp_shapes.append(aval.shape)
#                 tmp_uuids.append(ary_uuid[0])
#                 tmp_indices.append(indices)
#                 tmp_mesh_ids.append(mesh.mesh_id)
#                 tmp_meshes.append(mesh)
#                 tmp_arrays.append(array)
#             flat_shapes.append(tuple(tmp_shapes))
#             flat_uuids.append(tuple(tmp_uuids))
#             flat_indices.append(tuple(tmp_indices))
#             flat_mesh_ids.append(tuple(tmp_mesh_ids))
#             flat_arrays.append(
#                 ReplicatedDistributedArray(tmp_meshes, tmp_arrays))
#
#     for m in executable.mesh_group.meshes:
#         for w in m.workers:
#             w.load_opt_params_worker_func.remote(path, prefix_to_flat_idx,
#                                                  config, flat_shapes,
#                                                  flat_uuids, flat_indices,
#                                                  flat_mesh_ids)
#
#     return flat_arrays
#
#
# def init_cache_dis_array(executable, config, batch_size, dummy=False):
#     """Initialize cache with distributed arrays."""
#     cache = init_cache_np(config, batch_size)
#     alpa.global_config.use_dummy_value_for_benchmarking = dummy
#     _, batch_info = executable.get_input_placement_specs()
#     flat_args, in_tree = tree_flatten(cache)
#     flat_info = tree_leaves(batch_info["cache"])
#     if hasattr(executable, "mesh_group"):
#         ret = executable.mesh_group.shard_args_to_arrays(flat_info, flat_args)
#     else:
#         ret = executable.physical_mesh.shard_args_to_arrays_ps(
#             flat_info, flat_args)
#     alpa.global_config.use_dummy_value_for_benchmarking = False
#     return ret
