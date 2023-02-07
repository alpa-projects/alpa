"""CodeGen model implementation."""
import dataclasses
from dataclasses import dataclass
from functools import partial
import itertools
import math
import os
from typing import Callable, Optional, Tuple, Dict, Sequence

import alpa
from alpa.device_mesh import (DistributedArray, ReplicatedDistributedArray,
                              MeshHostWorker, create_remote_array_refs)
from alpa.model.model_util import ModelOutput
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
import flax.linen as nn
from flax.linen import combine_masks, dot_product_attention_weights, make_causal_mask
import jax
import flax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
from jax.interpreters import pxla
import jaxlib.xla_extension as jax_xla
import numpy as np
import ray
import torch
from tqdm import tqdm
from warnings import warn

from llm_serving.model.opt_model import init_cache_aval, init_mask_aval

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
}


@flax.struct.dataclass
class CodeGenModelOutput(ModelOutput):
    last_hidden_state: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@flax.struct.dataclass
class CodeGenLMOutput(ModelOutput):
    logits: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@dataclass(frozen=True)
class CodeGenConfig:
    pad: int = 1
    vocab_size: int = 50400
    max_seq_len: int = 2048
    n_ctx: int = 2048
    hidden_size: int = 4096
    num_hidden_layers: int = 28
    n_head: int = 16
    rotary_dim: int = 64
    n_inner: int = None
    activation_fn: str = 'gelu_new'
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    # Added
    decoder_input_dim: int = 4096
    decoder_ffn_embed_dim: int = 16384
    dtype: any = jnp.float16
    num_pp_stages: int = None
    tie_word_embeddings: bool = False
    use_cache: bool = True
    # parallelize
    mark_boundary: bool = True


# Copied from transformers.models.gptj.modeling_flax_gptj.create_sinusoidal_positions
def create_sinusoidal_positions(num_pos, dim):
    inv_freq = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
    sinusoid_inp = np.einsum("i , j -> i j", np.arange(num_pos), inv_freq).astype("float32")
    sin, cos = np.sin(sinusoid_inp), np.cos(sinusoid_inp)

    sentinel = dim // 2 + dim % 2
    out = np.zeros((num_pos, dim))
    out[:, 0:sentinel] = sin
    out[:, sentinel:] = cos

    return jnp.array(out, dtype=jnp.float16)

# Copied from transformers.models.gptj.modeling_flax_gptj.rotate_every_two
def rotate_every_two(tensor):
    rotate_half_tensor = jnp.stack((-tensor[:, :, :, 1::2], tensor[:, :, :, ::2]), axis=-1)
    rotate_half_tensor = rotate_half_tensor.reshape(rotate_half_tensor.shape[:-2] + (-1,))
    return rotate_half_tensor

# Copied from transformers.models.gptj.modeling_flax_gptj.apply_rotary_pos_emb
def apply_rotary_pos_emb(tensor, sincos):
    sin_pos, cos_pos = sincos
    sin_pos = sin_pos[:, :, None, :].repeat(2, 3)
    cos_pos = cos_pos[:, :, None, :].repeat(2, 3)
    return (tensor * cos_pos) + (rotate_every_two(tensor) * sin_pos)

class CodeGenAttention(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.n_head != 0:
            raise ValueError(
                f"`hidden_size`: {self.config.hidden_size} has to be a "
                f"multiple of `n_head`: {self.config.n_head}"
            )

        self.embed_dim = self.config.hidden_size
        self.head_dim = self.config.hidden_size // self.config.n_head
        self.rotary_dim = self.config.rotary_dim

        self.qkv_combined = nn.Dense(
            self.config.hidden_size * 3,
            dtype=self.dtype,
            use_bias=False
        )
        
        self.out_proj = nn.Dense(self.config.hidden_size, dtype=self.dtype, use_bias=False)
        self.resid_dropout = nn.Dropout(rate=self.config.resid_pdrop)

        pos_embd_dim = self.rotary_dim or self.embed_dim
        self.embed_positions = create_sinusoidal_positions(self.config.max_seq_len, pos_embd_dim)

    def _split_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.n_head, self.head_dim))

    def _merge_heads(self, hidden_states):
        return hidden_states.reshape(hidden_states.shape[:2] + (self.config.hidden_size,))

    def __call__(self,
                 hidden_states,
                 position_ids,
                 output_attentions: bool = False,
                 attention_cache=None,
                 attention_mask=None,
                 deterministic:bool = True):

        batch_size = hidden_states.shape[0]
        seq_length = hidden_states.shape[1]
        fused_qkv = self.qkv_combined(hidden_states)
        mp_num = 4 # number of cores on their TPU
        qkv_split = fused_qkv.reshape(fused_qkv.shape[:-1] + (mp_num, -1))
        query, value, key = jnp.split(qkv_split, 3, axis=-1)
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)
        key_length = attention_mask.shape[-1]
        causal_attention_mask = make_causal_mask(jnp.ones((batch_size, key_length)), dtype="bool")

        expanded = jax.nn.one_hot(position_ids, self.embed_positions.shape[0], dtype=self.dtype)
        sincos = expanded @ jnp.asarray(self.embed_positions, self.dtype)
        sincos = jnp.split(sincos, 2, axis=-1)
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sincos)
            q_rot = apply_rotary_pos_emb(q_rot, sincos)

            key = jnp.concatenate([k_rot, k_pass], axis=-1)
            query = jnp.concatenate([q_rot, q_pass], axis=-1)
        else:
            key = apply_rotary_pos_emb(key, sincos)
            query = apply_rotary_pos_emb(query, sincos)
            
        # for fast decoding causal attention mask should be shifted
        if attention_cache:
            causal_attention_mask_shift = attention_cache[2][0]
        else:
            causal_attention_mask_shift = 0

        if attention_cache:
            max_decoder_length = attention_cache[0].shape[1]
            causal_attention_mask = jax.lax.dynamic_slice(
                causal_attention_mask,
                (0, 0, causal_attention_mask_shift, 0),
                (1, 1, seq_length, max_decoder_length)
            )

            # Handle a special kind of internal padding added by alpa.
            # Note that this kind of internal padding is different from
            # the padding added by the tokenizer. This internal padding
            # should not update cache and step_ct
            # shape: [B, 1, 1, S_max]
            is_internal_padding = (attention_mask == 2)
            num_internal_pad = jnp.sum(is_internal_padding, axis=3).reshape(-1)
            attention_mask = (attention_mask == 1)

        attention_mask = combine_masks(attention_mask, causal_attention_mask)

        if attention_cache:
            cache_key, cache_value, cache_index = attention_cache
            *batch_dims, max_length, num_heads, depth_per_head = cache_key.shape
            # update key, value caches with our new 1d spatial slices
            cur_index = cache_index[0]
            indices = (0,) * len(batch_dims) + (cur_index, 0, 0)
            key = lax.dynamic_update_slice(cache_key, key, indices)
            value = lax.dynamic_update_slice(cache_value, value, indices)
            cache_key = key
            cache_value = value
            num_updated_cache_vectors = query.shape[1]
            # A line added from bloom_model
            attention_cache = key, value, cache_index + num_updated_cache_vectors - num_internal_pad
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

        attn_weights = dot_product_attention_weights(
            query,
            key,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attn_pdrop,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights, value)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output, deterministic=deterministic)

        outputs = (attn_output, attention_cache,
                   attn_weights) if output_attentions else (attn_output,
                                                            attention_cache)
        return outputs


class CodeGenBlock(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        hidden_size = self.config.hidden_size

        self.self = CodeGenAttention(self.config, dtype=self.dtype)
        self.mlp = CodeGenMLP(self.config)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 position_ids = None,
                 deterministic: bool = True,
                 output_attentions: bool = False,
                 attention_cache=None,
                 attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.self(hidden_states,
                                 position_ids=position_ids,
                                 output_attentions=output_attentions,
                                 attention_cache=attention_cache,
                                 attention_mask=attention_mask)
        attn_output = attn_outputs[0]
        attention_cache = attn_outputs[1]
        
        feed_forward_hidden_states = self.mlp(hidden_states, deterministic=deterministic)
        hidden_states = attn_output + feed_forward_hidden_states + residual
        outputs = (hidden_states, attention_cache)

        if output_attentions:
            outputs += (attn_outputs[2],)

        return outputs


class CodeGenMLP(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)

        self.fc_in = nn.Dense(
            4 * self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=kernel_init
        )
        self.fc_out = nn.Dense(
            self.config.hidden_size,
            dtype=self.dtype,
            kernel_init=kernel_init
        )
        self.act = ACT2FN[self.config.activation_fn]
        self.dropout = nn.Dropout(self.config.resid_pdrop)

    def __call__(self,
                 hidden_states,
                 deterministic: bool = True):
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        return hidden_states

class CodeGenTransformerLayerCollection(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.layers = [ 
            CodeGenBlock(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
        attention_mask=None
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        new_attention_cache = () if attention_cache is not None else None

        if self.config.num_pp_stages is not None:
            if self.config.num_hidden_layers % self.config.num_pp_stages != 0:
                warn("The number of hidden layers is not divisible by the number of stages")
            layers_per_stage = self.config.num_hidden_layers // self.config.num_pp_stages

        for i, layer in enumerate(self.layers):
            if self.config.num_pp_stages is not None:
                if i % layers_per_stage == 0 and i != 0:
                    stage_id = i // layers_per_stage
                    if self.config.mark_boundary and i // layers_per_stage < self.config.num_pp_stages:
                        mark_pipeline_boundary()

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_attention_cache = None
            if attention_cache is not None:
                layer_attention_cache = attention_cache[i]
            layer_outputs = layer(hidden_states,
                                  position_ids=position_ids,
                                  output_attentions=output_attentions,
                                  attention_cache=layer_attention_cache,
                                  attention_mask=attention_mask)
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

        return CodeGenModelOutput(last_hidden_state=hidden_states,
                              hidden_states=all_hidden_states,
                              attentions=all_attentions,
                              attention_cache=new_attention_cache)


class CodeGenTransformerModule(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.wte = nn.Embed(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.dtype
        )

        self.drop = nn.Dropout(rate=self.config.embd_pdrop)

        self.encoder = CodeGenTransformerLayerCollection(self.config,
                                                     dtype=self.dtype)

        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                           dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        position_ids,
        deterministic:bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
        attention_mask=None
    ):
        input_embeds = self.wte(input_ids.astype("i4"))
        
        hidden_states = self.drop(input_embeds, deterministic=deterministic)

        outputs = self.encoder(
            hidden_states,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_cache=attention_cache,
            attention_mask=attention_mask
        )
        hidden_states = outputs[0]
        hidden_states = self.layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = outputs.hidden_states + (hidden_states,)
            outputs = CodeGenModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=outputs.attentions, attention_cache=outputs.attention_cache)
        else:
            outputs = CodeGenModelOutput(last_hidden_state=hidden_states, hidden_states=outputs.hidden_states, attentions=outputs.attentions, attention_cache=outputs.attention_cache)

        if not return_dict:
            return (hidden_states,) + outputs[1:]

        return outputs


class CodeGenForLMModule(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.transformers = CodeGenTransformerModule(config=self.config,
                                                 dtype=self.dtype)

        self.lm_head = nn.Dense(
            self.config.vocab_size,
            dtype=jnp.float32,
            kernel_init=jax.nn.initializers.normal(stddev=self.config.initializer_range),
        )

    def __call__(
        self,
        input_ids,
        position_ids,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_cache=None,
        attention_mask=None
    ):
        # Model
        outputs = self.transformers(
            input_ids=input_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_cache=attention_cache,
            attention_mask=attention_mask
        )

        hidden_states = outputs[0]

        if self.config.tie_word_embeddings:
            shared_kernel = self.transformers.variables["params"]["wte"]["embedding"].T
            logits = self.lm_head.apply({"params": {"kernel": shared_kernel}}, hidden_states)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute the prediction scores
        if not return_dict:
            return (logits,) + outputs[1:]

        return CodeGenLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )

def get_config(name, **kwargs):
    if name in ["codegen-350m-mono", "codegen-350m-multi", "codegen-350m-nl"]:
        config = CodeGenConfig(
            max_seq_len=2048, num_hidden_layers=20, n_head=16,
            hidden_size=1024, decoder_input_dim=1024, decoder_ffn_embed_dim=1024 * 4,
            rotary_dim=32, bos_token_id=1, vocab_size=51200
        )
    elif name in ["codegen-2b-mono", "codegen-2b-multi", "codegen-2b-nl"]:
        config = CodeGenConfig(
            max_seq_len=2048, num_hidden_layers=32, n_head=32,
            hidden_size=2560, decoder_input_dim=2560, decoder_ffn_embed_dim=2560 * 4,
            rotary_dim=64, bos_token_id=1, vocab_size=51200
        )
    elif name in ["codegen-6b-mono", "codegen-6b-multi", "codegen-6b-nl"]:
        config = CodeGenConfig(
            max_seq_len=2048, num_hidden_layers=33, n_head=16,
            hidden_size=4096, decoder_input_dim=4096, decoder_ffn_embed_dim=4096 * 4,
            rotary_dim=64, bos_token_id=1, vocab_size=51200
        )
    elif name in ["codegen-16b-mono", "codegen-16b-multi", "codegen-16b-nl"]:
        config = CodeGenConfig(
            max_seq_len=2048, num_hidden_layers=34, n_head=24,
            hidden_size=6144, decoder_input_dim=6144, decoder_ffn_embed_dim=6144 * 4,
            rotary_dim=64, bos_token_id=1, vocab_size=51200
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)

def init_model_aval(config):
    """Initialize model with parameters with abstract values (shape-only arrays)."""
    model = CodeGenForLMModule(config, dtype=config.dtype)
    rngkey = jax.core.ShapedArray((2,), jnp.uint32)
    input_ids = jax.core.ShapedArray((1, 2), jnp.int32)
    position_ids = jax.core.ShapedArray((1, 2), jnp.int32)
    attention_mask = jax.core.ShapedArray((1, 1, 1, 2), jnp.int32)
    params = jax.eval_shape(model.init, rngkey, input_ids, position_ids, attention_mask=attention_mask)
    params = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, config.dtype),
                          params)
    return model, params

def init_cache_np(config, batch_size):
    """Init cache with numpy arrays."""
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
    head_dim = config.hidden_size // config.n_head

    all_cache = []
    for i in range(config.num_hidden_layers):
        layer_cache = (
            np.zeros((batch_size, config.max_seq_len,
                      config.n_head, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size, config.max_seq_len,
                      config.n_head, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size,), np.int32),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)

def inference_step_no_cache(params, batch, apply_func):
    logits = apply_func(params, batch["input_ids"], batch["position_ids"])[0]
    return logits


def load_params_np(params, path, config, dummy=False):
    """Load parameters with numpy arrays."""
    if dummy:
        np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
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
    load_param("params.transformers.layer_norm.scale", load_array("ln_f.weight"))
    load_param("params.transformers.layer_norm.bias", load_array("ln_f.bias"))
    load_param("params.transformers.wte.embedding", load_array("wte.weight"))
    load_param("params.lm_head.bias", load_array("lm_head.bias"))
    load_param("params.lm_head.kernel", load_array("lm_head.weight").transpose())

    for i in tqdm(range(config.num_hidden_layers)):
        param_prefix = f"params.transformers.encoder.{i}."
        load_prefix = f"h.{i}."
        # Attention weights
        load_param(
            param_prefix + "self.out_proj.kernel",
            load_array(load_prefix + "attn.out_proj.weight").transpose())
        load_param(
            param_prefix + "self.qkv_combined.kernel",
            load_array(load_prefix + "attn.qkv_proj.weight").transpose())

        load_param(param_prefix + "layer_norm.scale",
                   load_array(load_prefix + "ln_1.weight"))
        load_param(param_prefix + "layer_norm.bias",
                   load_array(load_prefix + "ln_1.bias"))

        # MLP weights
        load_param(param_prefix + "mlp.fc_in.kernel",
                   load_array(load_prefix + "mlp.fc_in.weight").transpose())
        load_param(param_prefix + "mlp.fc_in.bias",
                   np.transpose(load_array(load_prefix + "mlp.fc_in.bias")))
        load_param(param_prefix + "mlp.fc_out.bias",
                   load_array(load_prefix + "mlp.fc_out.bias"))
        load_param(param_prefix + "mlp.fc_out.kernel",
                   load_array(load_prefix + "mlp.fc_out.weight").transpose())

    return flax.core.freeze(params)

def get_jax_executable(config: CodeGenConfig,
                       encoder_chunk_sizes: Sequence[int],
                       output_attentions: bool = False,
                       output_hidden_states:bool = False):
    """Get a single-gpu executable."""
    model, params = init_model_aval(config)

    @jax.jit
    def inference_step(params, batch):
        output = model.apply(params,
                             input_ids=batch["input_ids"],
                             position_ids=batch["position_ids"],
                             attention_cache=batch["cache"],
                             attention_mask=batch["mask"],
                             output_attentions=output_attentions,
                             output_hidden_states=output_hidden_states)
        return output

    executables = {}
    for length in encoder_chunk_sizes:
        executables[length] = inference_step
    return executables, params


def get_pipeshard_executable(config: CodeGenConfig,
                             batch_size: int,
                             encoder_chunk_sizes: Sequence[int],
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
    
    def inference_step_with_cache(params, batch):
        output = model.apply(
            params,
            batch["input_ids"],
            batch["position_ids"],
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
                "position_ids":
                    jax.core.ShapedArray((batch_size, 1), jnp.int32),
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
    for seq_len in encoder_chunk_sizes:
        executable = alpa.parallelize(
            inference_step_with_cache,
            batch_argnums=(1,),
            method=method_with_input_sharding).get_executable(
                params, {
                    "input_ids":
                        jax.core.ShapedArray(
                            (batch_size, seq_len), jnp.int32),
                    "position_ids":
                        jax.core.ShapedArray(
                            (batch_size, seq_len), jnp.int32),
                    "cache":
                        cache,
                    "mask":
                        mask,
                })
        executable.dump_debug_info("tmp_executable_%d" % seq_len)
        executables[seq_len] = executable
    return executables, params


def load_codegen_params_worker_func(self, path, prefix_to_idx, config, shapes,
                                uuids, indices, mesh_ids):
    """The worker function to load CodeGen parameters."""

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array, is_position_embedding=False):
        i = prefix_to_idx[param_key]

        for j in range(len(mesh_ids[i])):
            if self.mesh_id != mesh_ids[i][j]:
                # print(f"skipping {param_key} on mesh {self.mesh_id} which is on  {mesh_ids[i][j]} and {uuids[i][j]}")
                continue
            
            if not is_position_embedding:
                assert shapes[i][j] == loaded_array.shape, (
                    f"{shapes[i][j]} vs. {loaded_array.shape}")
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

    layers_per_stage = config.num_hidden_layers // config.num_pp_stages

    load_param("params.transformers.layer_norm.scale", load_array("ln_f.weight"))
    load_param("params.transformers.layer_norm.bias", load_array("ln_f.bias"))
    load_param("params.transformers.wte.embedding", load_array("wte.weight"))
    load_param("params.lm_head.bias", load_array("lm_head.bias"))
    load_param("params.lm_head.kernel", load_array("lm_head.weight").transpose())
    
    for i in range(config.num_hidden_layers):
        stage_id = i // layers_per_stage
        if i // layers_per_stage  == config.num_pp_stages: # special case for codegen-6b
            stage_id = config.num_pp_stages - 1
        if stage_id != self.mesh_id:
            continue

        param_prefix = f"params.transformers.encoder.{i}."
        load_prefix = f"h.{i}."
        # Attention weights
        load_param(
            param_prefix + "self.out_proj.kernel",
            load_array(load_prefix + "attn.out_proj.weight").transpose())
        load_param(
            param_prefix + "self.qkv_combined.kernel",
            load_array(load_prefix + "attn.qkv_proj.weight").transpose())

        load_param(param_prefix + "layer_norm.scale",
                   load_array(load_prefix + "ln_1.weight"))
        load_param(param_prefix + "layer_norm.bias",
                   load_array(load_prefix + "ln_1.bias"))

        # MLP weights
        load_param(param_prefix + "mlp.fc_in.kernel",
                   load_array(load_prefix + "mlp.fc_in.weight").transpose())
        load_param(param_prefix + "mlp.fc_in.bias",
                   np.transpose(load_array(load_prefix + "mlp.fc_in.bias")))
        load_param(param_prefix + "mlp.fc_out.bias",
                   load_array(load_prefix + "mlp.fc_out.bias"))
        load_param(param_prefix + "mlp.fc_out.kernel",
                   load_array(load_prefix + "mlp.fc_out.weight").transpose())


setattr(MeshHostWorker, "load_codegen_params_worker_func",
        load_codegen_params_worker_func)


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
            w.load_codegen_params_worker_func.remote(path, prefix_to_flat_idx,
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
