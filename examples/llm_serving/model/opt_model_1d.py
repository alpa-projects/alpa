import heapq
import math
import queue
import time
import logging

import torch
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List, Union

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jaxlib.xla_extension as jax_xla
import numpy as np
import os
from enum import Enum
from functools import partial

from alpa.model.model_util import ModelOutput
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.util import OrderedSet
from alpa.timer import timers
from examples.llm_serving.model.opt_utils import sync


try:
    from ft_mha import fused_mmha, init_cache_manager, \
        prepare_inputs, free_cache, can_allocate
    from ft_mha import Prompt as PromptInternal, DecodingToken as DecodingTokenInternal
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


@flax.struct.dataclass
class OPTLMOutput(ModelOutput):
    logits: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None


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

        # Shape of cache_key and cache_value: [batch * max_length, heads, head_dim]
        # Shape of cache_index: [batch * max_length]
        cache_key, cache_value = attention_cache

        attn_output = fused_mmha(qkv_combined_states, self.qkv_combined_bias,
                                 cache_key, cache_value)

        attn_output = attn_output.reshape(attn_output.shape[:1] + (-1,))

        if output_attentions:
            print("Do not support output_attentions")
        return attn_output


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
        hidden_states = self.dense(attn_outputs)
        hidden_states = hidden_states + residual

        return hidden_states


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

        hidden_states = self.ffn(attention_outputs)
        return hidden_states


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
        all_hidden_states = () if output_hidden_states else None

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
            hidden_states = layer(hidden_states,
                                  output_attentions=output_attentions,
                                  attention_cache=layer_attention_cache)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return OPTModelOutput(last_hidden_state=hidden_states,
                              hidden_states=all_hidden_states)


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
            hidden_states=outputs.hidden_states)


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
            hidden_states=outputs.hidden_states)


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
            jax.core.ShapedArray((total_cache_len * config.n_head * head_dim,),
                                 dtype),
            jax.core.ShapedArray((total_cache_len * config.n_head * head_dim,),
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
            np.zeros((total_cache_len * config.n_head * head_dim),
                     dtype=np_dtype),
            np.zeros((total_cache_len * config.n_head * head_dim),
                     dtype=np_dtype),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def build_position_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).astype(np.int32)
    position_ids = np.cumsum(mask).astype(np.int32) * mask + padding_idx
    return position_ids


class PromptStatus(Enum):
    PROMPT = 1
    DECODING = 2
    FINISHED = 3


class Prompt:
    def __init__(self, input_ids, sentence_id, max_length=2048):
        self.input_ids = input_ids
        self.sentence_id = sentence_id
        self.status = PromptStatus.PROMPT
        # states to be filled during generation
        self.generated_ids = []
        self.last_generated_id = None

        # In v3, we have to use an internal Prompt object.
        self.p = PromptInternal(seq_id=sentence_id,
                                max_len=max_length,
                                token_ids=self.input_ids)
        # latency information
        self.start_time = None
        self.finish_time = None

    def finish(self, finish_token_id):
        self.finish_time = time.time()
        self.status = PromptStatus.FINISHED
        self.generated_ids.append(finish_token_id)
        self.last_generated_id = finish_token_id

    def add_token(self, token_id):
        if self.status == PromptStatus.PROMPT:
            self.status = PromptStatus.DECODING
        else:
            assert self.last_generated_id is not None and self.status == PromptStatus.DECODING
            self.generated_ids.append(self.last_generated_id)
        self.last_generated_id = token_id
        # rewrite the internal object to DecodingToken
        self.p = DecodingTokenInternal(seq_id=self.sentence_id, token_id=token_id)

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
    def latency(self):
        if self.status != PromptStatus.FINISHED:
            raise RuntimeError("Unfinished prompt.")
        return self.finish_time - self.start_time

    def print(self):
        print(self.input_ids + ":" + self.generated_ids)


class IterationLevelInputPool:
    """This pool is for iteration-level scheduling."""
    def __init__(self,
                 input_pool_config,
                 model_config,
                 max_length=None,
                 max_new_tokens=None):
        self.batch_size = input_pool_config.batch_size
        self.cache_size = input_pool_config.cache_size
        self.model_config = model_config
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens

        # Cache space is associated and owned with Pool.
        self.cache = jax.tree_map(jnp.array, init_cache_np(model_config, self.cache_size))
        init_cache_manager(cache_size=self.cache_size)

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
        """Enter a new batch of prompts into self."""
        sentence_ids = self.next_sentence_id(len(input_sequences))

        def max_new_tokens(seq_len):
            n = 2048
            if self.max_length:
                n = min(n, self.max_length - seq_len)
            if self.max_new_tokens:
                n = min(n, self.max_new_tokens)
            return n

        for i, seq in enumerate(input_sequences):
            p = Prompt(seq, sentence_ids[i], max_length=max_new_tokens(len(seq)) + len(seq))
            self.todo.put(p)

    def next(self):
        """Get the inputs for the next iteration from the pool."""
        # figure out WIP prompts and put their next token in a list
        decoding_input = list(self.wip)
        # re-batch new prompts, concat them into a list

        prompt_input = []
        proposals = []
        batch_availability = self.batch_size - len(decoding_input)
        while not self.todo.empty():
            proposals.append(self.todo.queue[0])
            proposals_length = [p.prompt_length for p in proposals]
            num_new_tokens = sum(proposals_length)
            # now we check if we can put this prompt into batch
            if batch_availability < num_new_tokens:
                break
            if not can_allocate(proposals_length):
                break
            prompt_input.append(self.todo.get())
        logger.debug(f"In this iteration {len(prompt_input)} new prompts enter.")

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
            position_ids.extend([i for i in range(start_idx, start_idx + p.prompt_length)])
        for p in decoding_input:
            start_idx = 1 + self.pad + p.num_prev_tokens
            position_ids.extend([start_idx])
        position_ids = np.array(position_ids +  [0] * (self.batch_size - len(position_ids)), dtype=np.int32)

        self._current_batch = prompt_input + decoding_input
        logit_positions = []
        i = -1
        for p in prompt_input:
            i += p.prompt_length
            logit_positions.append(i)
        for _ in decoding_input:
            i += 1
            logit_positions.append(i)

        # start prompts for recording time
        for p in prompt_input:
            p.start()

        # Call prepare_inputs before every inference_step.
        prepare_inputs([prompt.p for prompt in prompt_input], [prompt.p for prompt in decoding_input])
        # return inputs
        return input, input_index, position_ids, logit_positions

    def update(self, generated_ids):
        """Update the pool after one iteration of inference."""
        if self._current_batch is None:
            raise RuntimeError("There is no pending batch so update() is unnecessary.")
        for generated_id, p in zip(generated_ids, self._current_batch):
            # check EOS, move finished sentences from wip to finished queue
            if self.check_exit_condition(p, generated_id):
                if p.status == PromptStatus.DECODING:
                    assert p in self.wip
                    self.wip.remove(p)
                exit_reason = "EOS" if generated_id == self.eos else "reaching max length"
                logger.debug(f"Prompt {p.sentence_id} exits because of {exit_reason}. ")
                p.finish(generated_id)
                free_cache(p.sentence_id)
                self.done.add(p)
            elif p.status == PromptStatus.PROMPT:
                # PROMPT -> DECODING
                p.add_token(generated_id)
                self.wip.add(p)
            elif p.status == PromptStatus.DECODING:
                # DECODING -> DECODING
                p.add_token(generated_id)
            else:
                raise RuntimeError(f"Prompt status: {p.status} should not appear here." )

    def get_results(self):
        """Return results sorted by their sentence id."""
        sorted_results = sorted(self.done, key=lambda x: x.sentence_id, reverse=False)
        return [p.input_ids + p.generated_ids for p in sorted_results]

    def get_latency(self):
        """Return the latency of each prompt following their sequence id."""
        sorted_results = sorted(self.done, key=lambda x: x.sentence_id, reverse=False)
        return [p.latency for p in sorted_results]

    def next_sentence_id(self, number):
        counter = self._sentence_id_counter
        if number == 1:
            ret = [counter]
        else:
            ret = list(range(counter, counter + number))
        self._sentence_id_counter = (counter + number) % (1 << 60)
        return ret

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
                             )
        return output.logits

    # executables = {}
    # for length in encoder_chunk_sizes:
    #     executables[length] = inference_step
    return inference_step, params
