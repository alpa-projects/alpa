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
import jax
import flax
from jax import lax
import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves
from jax.interpreters import pxla
import jaxlib.xla_extension as jax_xla
import numpy as np
import ray
from tqdm import tqdm

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


# TODO(chris): figure out which config params need to be pruned
@dataclass(frozen=True)
class CodeGenConfig:
    # Inherited from CodeGen
    # decoder_layers: int = 20
    # max_target_positions: int = 2048
    # decoder_embed_dim: int = 768
    # decoder_attention_heads: int = 12
    # decoder_input_dim: int = 768
    # decoder_ffn_embed_dim: int = 3072
    # pad: int = 1
    # activation_fn: str = 'relu'
    # dtype: any = jnp.float16
    # use_stable_embedding: bool = False
    # no_scale_embedding: bool = True
    # decoder_learned_pos: bool = True
    # share_decoder_input_output_embed: bool = True
    vocab_size: int = 50400
    max_target_positions: int = 2048
    n_ctx: int = 2048
    decoder_embed_dim: int = 4096
    decoder_layers: int = 28
    decoder_attention_heads: int = 16
    rotary_dim: int = 64
    n_inner: int = None
    activation_fn: str = 'gelu_new'
    resid_pdrop: float = 0.0
    embd_pdrop: float = 0.0
    attn_pdrop: float = 0.0
    layer_norm_eps: float = 0.00001
    initializer_range: float = 0.02
    scale_attn_weights: bool = True
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    # Added
    decoder_input_dim: int = 4096
    decoder_ffn_embed_dim: int = 16384
    dtype: any = jnp.float16
    num_pp_stages: int = None,
    # parallelize
    mark_boundary: bool = True


# Copied from transformers.models.marian.modeling_flax_marian.create_sinusoidal_positions
def create_sinusoidal_positions(n_pos, dim):
    position_enc = np.array([[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)])
    sentinel = dim // 2 + dim % 2
    out = np.zeros_like(position_enc)
    out[:, 0:sentinel] = np.sin(position_enc[:, 0::2])
    out[:, sentinel:] = np.cos(position_enc[:, 1::2])

    return jnp.array(out)

# TODO(chris) refactor: change embeddings to follow rotary position embeddings
class CodeGenEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert not self.config.use_stable_embedding
        self.embed_scale = 1.0 if self.config.no_scale_embedding else math.sqrt(
            self.config.decoder_embed_dim)
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.decoder_input_dim,
            dtype=self.dtype,
        )
        assert self.config.max_target_positions is not None
        assert self.config.decoder_learned_pos
        self.position_embeddings = nn.Embed(
            self.config.max_target_positions + self.config.pad + 1,
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        )
        self.project_in_dim = nn.Dense(
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        ) if self.config.decoder_input_dim != self.config.decoder_embed_dim else None

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


class CodeGenAttention(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        if self.config.decoder_embed_dim % self.config.decoder_attention_heads != 0:
            raise ValueError(
                f"`decoder_embed_dim`: {self.config.decoder_embed_dim} has to be a "
                f"multiple of `decoder_attention_heads`: {self.config.decoder_attention_heads}"
            )

        self.qkv_combined = nn.Dense(
            self.config.decoder_embed_dim * 3,
            dtype=self.dtype,
        )
        
        self.rotary_dim = self.config.rotary_dim

    def __call__(self,
                 hidden_states,
                 sinusoidal_pos,
                 output_attentions: bool = False,
                 attention_cache=None,
                 attention_mask=None):
        head_dim = self.config.decoder_embed_dim // self.config.decoder_attention_heads

        qkv_combined_states = self.qkv_combined(hidden_states)
        qkv_combined_states = qkv_combined_states.reshape(
            qkv_combined_states.shape[:2] + (-1, 3))
        query_states, key_states, value_states = jnp.split(qkv_combined_states,
                                                           3,
                                                           axis=3)

        # shape: [B, S, #head, head_dim]
        query_states = query_states.reshape(hidden_states.shape[:2] + (
            self.config.decoder_attention_heads, head_dim))
        # shape: [B, S, #head, head_dim]
        value_states = value_states.reshape(hidden_states.shape[:2] + (
            self.config.decoder_attention_heads, head_dim))
        # shape: [B, S, #head, head_dim]
        key_states = key_states.reshape(hidden_states.shape[:2] +
                                        (self.config.decoder_attention_heads,
                                         head_dim))
        

        if sinusoidal_pos is not None:
            if self.rotary_dim:
                query_states, key_states, value_states = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_states, key_states, value_states
                )
            else:
                query_states, key_states = self.apply_rotary_position_embeddings(
                    sinusoidal_pos, query_states, key_states
                )

        batch_size = hidden_states.shape[0]
        if attention_cache is None:
            query_len, key_len = query_states.shape[1], key_states.shape[1]
            assert query_len == key_len
            # shape: [B, 1, S_max, S_max]
            causal_mask = nn.make_causal_mask(
                jnp.ones((batch_size, key_len)), dtype="bool")
            # shape: [B, 1, 1, S_max]
            input_mask = attention_mask
            # shape: [B, 1, S_max, S_max]
            mask = nn.combine_masks(causal_mask, input_mask, dtype="bool")
        else:
            cache_key, cache_value, cache_index = attention_cache
            cache_index_ = cache_index[0]
            update_indices = (0, cache_index_, 0, 0)
            # shape: [B, S_max, #head, head_dim]
            key_states = lax.dynamic_update_slice(cache_key, key_states, update_indices)
            # shape: [B, S_max, #head, head_dim]
            value_states = lax.dynamic_update_slice(cache_value, value_states, update_indices)
            query_len, key_len = query_states.shape[1], key_states.shape[1]

            # Handle a special kind of internal padding added by alpa.
            # Note that this kind of internal padding is different from
            # the padding added by the tokenizer. This internal padding
            # should not update cache and step_ct
            # shape: [B, 1, 1, S_max]
            is_internal_padding = (attention_mask == 2)
            num_internal_pad = jnp.sum(is_internal_padding, axis=3).reshape(-1)
            attention_mask = (attention_mask == 1)

            attention_cache = key_states, value_states, cache_index + query_len - num_internal_pad

            # shape: [B, 1, S_max, S_max]
            causal_mask = nn.make_causal_mask(
                jnp.ones((batch_size, key_len)), dtype="bool")
            # shape: [B, 1, S, S_max]
            causal_mask = lax.dynamic_slice(causal_mask,
                (0, 0, cache_index_, 0), (batch_size, 1, query_len, key_len))
            # shape: [B, 1, 1, S_max]
            input_mask = attention_mask
            # shape: [B, 1, S, S_max]
            mask = nn.combine_masks(causal_mask, input_mask, dtype="bool")

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            mask=mask,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights,
                                 value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output, attention_cache,
                   attn_weights) if output_attentions else (attn_output,
                                                            attention_cache)
        return outputs


        
    # taken from https://github.com/huggingface/transformers/blob/80367cd1fb6d36ca6bdd99b70586aab4ffae1ae1/src/transformers/models/roformer/modeling_flax_roformer.py#L275
    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
        sin, cos = sinusoidal_pos.split(2, axis=-1)
        sin_pos = jnp.stack([sin, sin], axis=-1).reshape(sinusoidal_pos.shape)
        cos_pos = jnp.stack([cos, cos], axis=-1).reshape(sinusoidal_pos.shape)

        def rotate_layer(layer, sin_pos, cos_pos):
            rotate_half_layer = jnp.stack([-layer[..., 1::2], layer[..., ::2]], axis=-1).reshape(layer.shape)
            rotary_matrix_cos = jnp.einsum("bslh,...sh->bslh", layer, cos_pos)
            rotary_matrix_sin = jnp.einsum("bslh,...sh->bslh", rotate_half_layer, sin_pos)
            return rotary_matrix_cos + rotary_matrix_sin

        query_layer = rotate_layer(query_layer, sin_pos, cos_pos)
        key_layer = rotate_layer(key_layer, sin_pos, cos_pos)
        if value_layer is not None:
            value_layer = rotate_layer(value_layer, sin_pos, cos_pos)
            return query_layer, key_layer, value_layer
        return query_layer, key_layer


class CodeGenBlock(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16

    def setup(self):
        self.self = CodeGenAttention(self.config, dtype=self.dtype)
        self.mlp = CodeGenMLP(self.config)
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 sinusoidal_pos,
                 output_attentions: bool = False,
                 attention_cache=None,
                 attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.self(hidden_states,
                                 sinusoidal_pos=sinusoidal_pos,
                                 output_attentions=output_attentions,
                                 attention_cache=attention_cache,
                                 attention_mask=attention_mask)
        attn_output = attn_outputs[0]
        attention_cache = attn_outputs[1]
        
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + feed_forward_hidden_states + residual
        outputs = (hidden_states, attention_cache)

        if output_attentions:
            outputs += (attn_outputs[2],)

        return outputs


class CodeGenMLP(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.fc_in = nn.Dense(
            self.config.decoder_ffn_embed_dim,
            dtype=self.dtype,
        )
        self.act = ACT2FN[self.config.activation_fn]
        self.fc_out = nn.Dense(
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(self.config.resid_pdrop)

    def __call__(self, hidden_states):
        residual = hidden_states
        # TODO(chris): we are changing order here to follow codegen, double check this is correct
        hidden_states = self.fc_in(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.fc_out(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class CodeGenTransformerLayer(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert not getattr(self.config, "cross_self_attention", False)
        assert not getattr(self.config, "scale_heads", False)
        assert not getattr(self.config, "scale_attn", False)
        assert not getattr(self.config, "scale_fc", False)
        self.attention = CodeGenBlock(self.config, dtype=self.dtype)
        self.mlp = CodeGenMLP(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 sinusoidal_pos,
                 output_attentions: bool = False,
                 attention_cache=None,
                 attention_mask=None):

        attention_outputs = self.attention(hidden_states,
                                           sinusoidal_pos=sinusoidal_pos,
                                           output_attentions=output_attentions,
                                           attention_cache=attention_cache,
                                           attention_mask=attention_mask)
        attention_output = attention_outputs[0]
        attention_cache = attention_outputs[1]

        hidden_states = self.mlp(attention_output)

        outputs = (hidden_states, attention_cache)

        if output_attentions:
            outputs += (attention_outputs[2],)
        return outputs


class CodeGenTransformerLayerCollection(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.layers = [
            CodeGenTransformerLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
        ]

    def __call__(
        self,
        hidden_states,
        sinusoidal_pos,
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
            assert self.config.decoder_layers % self.config.num_pp_stages == 0
            layers_per_stage = self.config.decoder_layers // self.config.num_pp_stages

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
                                  sinusoidal_pos=sinusoidal_pos,
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
        self.embeddings = CodeGenEmbeddings(self.config, dtype=self.dtype)
        self.embed_positions = create_sinusoidal_positions(
            self.config.max_target_positions, self.config.decoder_embed_dim // self.config.decoder_attention_heads
        )
        self.encoder = CodeGenTransformerLayerCollection(self.config,
                                                     dtype=self.dtype)
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
        attention_mask=None
    ):
        hidden_states = self.embeddings(input_ids, position_ids)

        sinusoidal_pos = self.embed_positions[: hidden_states.shape[1], :]

        outputs = self.encoder(
            hidden_states,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            sinusoidal_pos=sinusoidal_pos,
            return_dict=return_dict,
            attention_cache=attention_cache,
            attention_mask=attention_mask
        )
        hidden_states = outputs[0]
        hidden_states = self.layer_norm(hidden_states)

        if not return_dict:
            # if pooled is None, don't return it
            return (hidden_states,) + outputs[1:]

        return CodeGenModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )


class CodeGenForLMModule(nn.Module):
    config: CodeGenConfig
    dtype: jnp.dtype = jnp.float16
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = CodeGenTransformerModule(config=self.config,
                                                 dtype=self.dtype)

        self.project_out_dim = nn.Dense(
            self.config.decoder_input_dim,
            dtype=self.dtype,
        ) if self.config.decoder_input_dim != self.config.decoder_embed_dim else None

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
        attention_mask=None
    ):
        # Model
        outputs = self.transformers(
            input_ids,
            position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            attention_cache=attention_cache,
            attention_mask=attention_mask
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

        return CodeGenLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )

# TODO(chris) refactor: replace with hyperparameters from the paper
#  - weight decay
def get_codegen_config(name, **kwargs):
    if name == "350M":
        config = CodeGenConfig(
            max_target_positions=2048, decoder_layers=20, decoder_attention_heads=16,
            decoder_embed_dim=1024, decoder_input_dim=1024, decoder_ffn_embed_dim=1024 * 4,
            rotary_dim=32, bos_token_id=1
        )
    elif name == "2.7B":
        config = CodeGenConfig(
            max_target_positions=2048, decoder_layers=32, decoder_attention_heads=32,
            decoder_embed_dim=2560, decoder_input_dim=2560, decoder_ffn_embed_dim=2560 * 4,
            rotary_dim=64, bos_token_id=1
        )
    elif name == "6.1B":
        config = CodeGenConfig(
            max_target_positions=2048, decoder_layers=33, decoder_attention_heads=16,
            decoder_embed_dim=4096, decoder_input_dim=4096, decoder_ffn_embed_dim=4096 * 4,
            rotary_dim=64, bos_token_id=1
        )
    elif name == "16.1B":
        config = CodeGenConfig(
            max_target_positions=2048, decoder_layers=34, decoder_attention_heads=24,
            decoder_embed_dim=6144, decoder_input_dim=6144, decoder_ffn_embed_dim=6144 * 4,
            rotary_dim=64, bos_token_id=1
        )
    else:
        raise ValueError(f"Invalid model name: {name}")

    return dataclasses.replace(config, **kwargs)

# TODO(chris): fix this to fit the same params as Codegen
def init_model_aval(config):
    """Initialize model with parameters with abstract values (shape-only arrays)."""
    model = CodeGenForLMModule(config, dtype=config.dtype)
    rngkey = jax.core.ShapedArray((2,), jnp.uint32)
    input_ids = jax.core.ShapedArray((1, 128), jnp.int32)
    position_ids = jax.core.ShapedArray((1, 128), jnp.int32)
    params = jax.eval_shape(model.init, rngkey, input_ids, position_ids)
    params = jax.tree_map(lambda x: jax.ShapeDtypeStruct(x.shape, config.dtype),
                          params)
    return model, params


def init_cache_aval(config, batch_size):
    """Initialize cache with abstract values (shape-only arrays)."""
    dtype = config.dtype
    head_dim = config.decoder_embed_dim // config.decoder_attention_heads

    all_cache = []
    for _ in range(config.decoder_layers):
        layer_cache = (
            jax.core.ShapedArray((batch_size, config.max_target_positions,
                                  config.decoder_attention_heads, head_dim),
                                 dtype),
            jax.core.ShapedArray((batch_size, config.max_target_positions,
                                  config.decoder_attention_heads, head_dim),
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
    np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
    head_dim = config.decoder_embed_dim // config.decoder_attention_heads

    all_cache = []
    for i in range(config.decoder_layers):
        layer_cache = (
            np.zeros((batch_size, config.max_target_positions,
                      config.decoder_attention_heads, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size, config.max_target_positions,
                      config.decoder_attention_heads, head_dim),
                     dtype=np_dtype),
            np.zeros((batch_size,), np.int32),
        )
        all_cache.append(layer_cache)
    return tuple(all_cache)


def build_position_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).astype(np.int32)
    position_ids = np.cumsum(mask, axis=1).astype(np.int32) * mask + padding_idx
    return position_ids


def inference_step_no_cache(params, batch, apply_func):
    logits = apply_func(params, batch["input_ids"], batch["position_ids"])[0]
    return logits


# TODO(chris): rename the loaded np arrays
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
    load_param("params.transformers.embeddings.word_embeddings.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("params.transformers.embeddings.position_embeddings.embedding",
               load_array("decoder.embed_positions.weight"),
               is_position_embedding=True)
    if config.version > 2:
        load_param("params.transformers.layer_norm.scale",
                   load_array("decoder.layer_norm.weight"))
        load_param("params.transformers.layer_norm.bias",
                   load_array("decoder.layer_norm.bias"))
    for i in tqdm(range(config.decoder_layers)):
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
        b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
            (3, dim)).transpose([1, 0]).reshape((-1,))
        load_param(param_prefix + "attention.self.qkv_combined.bias", b_qkv)
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


def get_jax_executable(config: CodeGenConfig,
                       encoder_chunk_sizes: Sequence[int],
                       output_attentions: bool = False,
                       output_hidden_states:bool = False):
    """Get a single-gpu executable."""
    model, params = init_model_aval(config)

    @jax.jit
    def inference_step(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
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
    #method = alpa.ShardParallel()

    if autoregressive:

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
    else:
        assert len(encoder_chunk_sizes) == 1
        seq_len = encoder_chunk_sizes[0]

        @alpa.parallelize(batch_argnums=(1,), method=method)
        def inference_step(params, batch):
            output = model.apply(
                params,
                batch["input_ids"],
                batch["position_ids"],
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
                "position_ids":
                    jax.core.ShapedArray(
                        (batch_size, seq_len), jnp.int32),
            })

        executable.dump_debug_info("tmp")
    return {seq_len: executable}, params


def load_opt_params_worker_func(self, path, prefix_to_idx, config, shapes,
                                uuids, indices, mesh_ids):
    """The worker function to load CodeGen parameters."""

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array, is_position_embedding=False):
        i = prefix_to_idx[param_key]

        for j in range(len(mesh_ids[i])):
            if self.mesh_id != mesh_ids[i][j]:
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

    load_param("params.transformers.embeddings.word_embeddings.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("params.transformers.embeddings.position_embeddings.embedding",
               load_array("decoder.embed_positions.weight"),
               is_position_embedding=True)

    if config.version > 2:
        load_param("params.transformers.layer_norm.scale",
                   load_array("decoder.layer_norm.weight"))
        load_param("params.transformers.layer_norm.bias",
                   load_array("decoder.layer_norm.bias"))

    layers_per_stage = config.decoder_layers // config.num_pp_stages

    for i in range(config.decoder_layers):
        stage_id = i // layers_per_stage
        if stage_id != self.mesh_id:
            continue

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
        b_qkv = np.concatenate([bq, bk, bv], axis=0).reshape(
            (3, dim)).transpose([1, 0]).reshape((-1,))
        load_param(param_prefix + "attention.self.qkv_combined.bias", b_qkv)
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


setattr(MeshHostWorker, "load_opt_params_worker_func",
        load_opt_params_worker_func)


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