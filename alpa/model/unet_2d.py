"""
This file is modified from multiple files in
https://github.com/huggingface/diffusers/blob/main/src/diffusers/models
"""

# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import math
from typing import Tuple, Union
import flax
import flax.linen as nn
import jax
from jax.experimental.maps import FrozenDict
import jax.numpy as jnp

from alpa import mark_pipeline_boundary
from alpa.model.bert_model import BertConfig
from alpa.model.model_util import ModelOutput


# FIXME: not from bert config
class UNet2DConfig(BertConfig):

    def __init__(self,
                 *,
                 sample_size: int = 32,
                 in_channels: int = 4,
                 out_channels: int = 4,
                 layers_per_block: int = 2,
                 freq_shift: int = 0,
                 num_groups: int = 4,
                 **kwargs):
        super().__init__(**kwargs)
        self.sample_size = sample_size
        self.in_channels = in_channels,
        self.out_channels = out_channels
        self.layers_per_block = layers_per_block
        self.freq_shift = freq_shift
        # Group Norm factor
        self.num_groups = num_groups


@flax.struct.dataclass
class FlaxUNet2DConditionOutput(ModelOutput):
    """
    Args:
        sample (`jnp.ndarray` of shape `(batch_size, num_channels, height, width)`):
            Hidden states conditioned on `encoder_hidden_states` input. Output of last layer of model.
    """

    sample: jnp.ndarray


##### Embeddings - Do not add pipeline marker at this level
def get_sinusoidal_embeddings(timesteps, embedding_dim, freq_shift: float = 1):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] tensor of positional embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - freq_shift)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.cos(emb), jnp.sin(emb)], -1)
    return emb


class FlaxTimestepEmbedding(nn.Module):
    r"""
    Time step Embedding Module. Learns embeddings for input time steps.
    Args:
        time_embed_dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
                Parameters `dtype`
    """
    time_embed_dim: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, temb):
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype,
                        name="linear_1")(temb)
        temb = nn.silu(temb)
        temb = nn.Dense(self.time_embed_dim, dtype=self.dtype,
                        name="linear_2")(temb)
        return temb


class FlaxTimesteps(nn.Module):
    r"""
    Wrapper Module for sinusoidal Time step Embeddings as described in https://arxiv.org/abs/2006.11239
    Args:
        dim (`int`, *optional*, defaults to `32`):
                Time step embedding dimension
    """
    dim: int = 32
    freq_shift: float = 1

    @nn.compact
    def __call__(self, timesteps):
        return get_sinusoidal_embeddings(timesteps,
                                         self.dim,
                                         freq_shift=self.freq_shift)


##### ResNetBlocks - Do not add pipeline marker at this level
class FlaxUpsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        batch, height, width, channels = hidden_states.shape
        hidden_states = jax.image.resize(
            hidden_states,
            shape=(batch, height * 2, width * 2, channels),
            method="nearest",
        )
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxDownsample2D(nn.Module):
    out_channels: int
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.conv = nn.Conv(
            self.out_channels,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),  # padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        # pad = ((0, 0), (0, 1), (0, 1), (0, 0))  # pad height and width dim
        # hidden_states = jnp.pad(hidden_states, pad_width=pad)
        hidden_states = self.conv(hidden_states)
        return hidden_states


class FlaxResnetBlock2D(nn.Module):
    in_channels: int
    config: UNet2DConfig
    out_channels: int = None
    use_nin_shortcut: bool = None
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        out_channels = (self.in_channels
                        if self.out_channels is None else self.out_channels)

        self.norm1 = nn.GroupNorm(num_groups=self.config.num_groups,
                                  epsilon=1e-5)
        self.conv1 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        self.time_emb_proj = nn.Dense(out_channels, dtype=self.dtype)

        self.norm2 = nn.GroupNorm(num_groups=self.config.num_groups,
                                  epsilon=1e-5)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.conv2 = nn.Conv(
            out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        use_nin_shortcut = (self.in_channels != out_channels
                            if self.use_nin_shortcut is None else
                            self.use_nin_shortcut)

        self.conv_shortcut = None
        if use_nin_shortcut:
            self.conv_shortcut = nn.Conv(
                out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                dtype=self.dtype,
            )

    def __call__(self, hidden_states, temb, deterministic=True):
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.conv1(hidden_states)

        temb = self.time_emb_proj(nn.swish(temb))
        temb = jnp.expand_dims(jnp.expand_dims(temb, 1), 1)
        hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)
        hidden_states = nn.swish(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return hidden_states + residual


##### Attentions - Do not add pipeline marker at this level
class FlaxAttentionBlock(nn.Module):
    r"""
    A Flax multi-head attention module as described in: https://arxiv.org/abs/1706.03762
    Parameters:
        query_dim (:obj:`int`):
            Input hidden states dimension
        heads (:obj:`int`, *optional*, defaults to 8):
            Number of heads
        dim_head (:obj:`int`, *optional*, defaults to 64):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    query_dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim_head * self.heads
        self.scale = self.dim_head**-0.5

        # Weights were exported with old names {to_q, to_k, to_v, to_out}
        self.query = nn.Dense(inner_dim,
                              use_bias=False,
                              dtype=self.dtype,
                              name="to_q")
        self.key = nn.Dense(inner_dim,
                            use_bias=False,
                            dtype=self.dtype,
                            name="to_k")
        self.value = nn.Dense(inner_dim,
                              use_bias=False,
                              dtype=self.dtype,
                              name="to_v")

        self.proj_attn = nn.Dense(self.query_dim,
                                  dtype=self.dtype,
                                  name="to_out_0")

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size,
                                dim // head_size)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size * head_size, seq_len,
                                dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len,
                                dim)
        tensor = jnp.transpose(tensor, (0, 2, 1, 3))
        tensor = tensor.reshape(batch_size // head_size, seq_len,
                                dim * head_size)
        return tensor

    def __call__(self, hidden_states, context=None, deterministic=True):
        context = hidden_states if context is None else context

        query_proj = self.query(hidden_states)
        key_proj = self.key(context)
        value_proj = self.value(context)

        query_states = self.reshape_heads_to_batch_dim(query_proj)
        key_states = self.reshape_heads_to_batch_dim(key_proj)
        value_states = self.reshape_heads_to_batch_dim(value_proj)

        # compute attentions
        attention_scores = jnp.einsum("b i d, b j d->b i j", query_states,
                                      key_states)
        attention_scores = attention_scores * self.scale
        attention_probs = nn.softmax(attention_scores, axis=2)

        # attend to values
        hidden_states = jnp.einsum("b i j, b j d -> b i d", attention_probs,
                                   value_states)
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.proj_attn(hidden_states)
        return hidden_states


class FlaxBasicTransformerBlock(nn.Module):
    r"""
    A Flax transformer block layer with `GLU` (Gated Linear Unit) activation function as described in:
    https://arxiv.org/abs/1706.03762
    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    n_heads: int
    d_head: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # self attention
        self.attn1 = FlaxAttentionBlock(self.dim,
                                        self.n_heads,
                                        self.d_head,
                                        self.dropout,
                                        dtype=self.dtype)
        # cross attention
        self.attn2 = FlaxAttentionBlock(self.dim,
                                        self.n_heads,
                                        self.d_head,
                                        self.dropout,
                                        dtype=self.dtype)
        self.ff = FlaxGluFeedForward(dim=self.dim,
                                     dropout=self.dropout,
                                     dtype=self.dtype)
        self.norm1 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm2 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)
        self.norm3 = nn.LayerNorm(epsilon=1e-5, dtype=self.dtype)

    def __call__(self, hidden_states, context, deterministic=True):
        # self attention
        residual = hidden_states
        hidden_states = self.attn1(self.norm1(hidden_states),
                                   deterministic=deterministic)
        hidden_states = hidden_states + residual

        # cross attention
        residual = hidden_states
        hidden_states = self.attn2(self.norm2(hidden_states),
                                   context,
                                   deterministic=deterministic)
        hidden_states = hidden_states + residual

        # feed forward
        residual = hidden_states
        hidden_states = self.ff(self.norm3(hidden_states),
                                deterministic=deterministic)
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxSpatialTransformer(nn.Module):
    r"""
    A Spatial Transformer layer with Gated Linear Unit (GLU) activation function as described in:
    https://arxiv.org/pdf/1506.02025.pdf
    Parameters:
        in_channels (:obj:`int`):
            Input number of channels
        n_heads (:obj:`int`):
            Number of heads
        d_head (:obj:`int`):
            Hidden states dimension inside each head
        depth (:obj:`int`, *optional*, defaults to 1):
            Number of transformers block
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    n_heads: int
    d_head: int
    depth: int = 1
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.norm = nn.GroupNorm(num_groups=32, epsilon=1e-5)

        inner_dim = self.n_heads * self.d_head
        self.proj_in = nn.Conv(
            inner_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

        self.transformer_blocks = [
            FlaxBasicTransformerBlock(inner_dim,
                                      self.n_heads,
                                      self.d_head,
                                      dropout=self.dropout,
                                      dtype=self.dtype)
            for _ in range(self.depth)
        ]

        self.proj_out = nn.Conv(
            inner_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            dtype=self.dtype,
        )

    def __call__(self, hidden_states, context, deterministic=True):
        batch, height, width, channels = hidden_states.shape
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)

        hidden_states = hidden_states.reshape(batch, height * width, channels)

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states,
                                              context,
                                              deterministic=deterministic)

        hidden_states = hidden_states.reshape(batch, height, width, channels)

        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class FlaxGluFeedForward(nn.Module):
    r"""
    Flax module that encapsulates two Linear layers separated by a gated linear unit activation from:
    https://arxiv.org/abs/2002.05202
    Parameters:
        dim (:obj:`int`):
            Inner hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # The second linear layer needs to be called
        # net_2 for now to match the index of the Sequential layer
        self.net_0 = FlaxGEGLU(self.dim, self.dropout, self.dtype)
        self.net_2 = nn.Dense(self.dim, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.net_0(hidden_states)
        hidden_states = self.net_2(hidden_states)
        return hidden_states


class FlaxGEGLU(nn.Module):
    r"""
    Flax implementation of a Linear layer followed by the variant of the gated linear unit activation function from
    https://arxiv.org/abs/2002.05202.
    Parameters:
        dim (:obj:`int`):
            Input hidden states dimension
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    dim: int
    dropout: float = 0.0
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        inner_dim = self.dim * 4
        self.proj = nn.Dense(inner_dim * 2, dtype=self.dtype)

    def __call__(self, hidden_states, deterministic=True):
        hidden_states = self.proj(hidden_states)
        hidden_linear, hidden_gelu = jnp.split(hidden_states, 2, axis=2)
        return hidden_linear * nn.gelu(hidden_gelu)


##### UNetBlocks - Add pipeline marker at this level
class FlaxCrossAttnDownBlock2D(nn.Module):
    r"""
    Cross Attention 2D Downsizing block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104
    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    config: UNet2DConfig
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        attentions = []

        for i in range(self.config.layers_per_block):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                config=self.config,
                out_channels=self.out_channels,
                dtype=self.dtype,
            )
            resnets.append(res_block)

            attn_block = FlaxSpatialTransformer(
                in_channels=self.out_channels,
                n_heads=self.config.num_attention_heads,
                d_head=self.out_channels // self.config.num_attention_heads,
                depth=1,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions

        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels,
                                                   dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 temb,
                 encoder_hidden_states,
                 deterministic=True):
        output_states = ()

        for idx, (resnet, attn) in enumerate(zip(self.resnets,
                                                 self.attentions)):
            hidden_states = resnet(hidden_states,
                                   temb,
                                   deterministic=deterministic)
            hidden_states = attn(hidden_states,
                                 encoder_hidden_states,
                                 deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                if idx != self.config.layers_per_block - 1:
                    mark_pipeline_boundary()
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)
        if self.config.add_manual_pipeline_markers:
            mark_pipeline_boundary()

        return hidden_states, output_states


class FlaxDownBlock2D(nn.Module):
    r"""
    Flax 2D downsizing block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        config (:obj:`UNet2DConfig`):
            UNet Global Config
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    config: UNet2DConfig
    add_downsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []

        for i in range(self.config.layers_per_block):
            in_channels = self.in_channels if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=in_channels,
                config=self.config,
                out_channels=self.out_channels,
                dtype=self.dtype,
            )
            resnets.append(res_block)
        self.resnets = resnets

        if self.add_downsample:
            self.downsamplers_0 = FlaxDownsample2D(self.out_channels,
                                                   dtype=self.dtype)

    def __call__(self, hidden_states, temb, deterministic=True):
        output_states = ()

        for idx, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states,
                                   temb,
                                   deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                if idx != self.config.layers_per_block - 1:
                    mark_pipeline_boundary()
            output_states += (hidden_states,)

        if self.add_downsample:
            hidden_states = self.downsamplers_0(hidden_states)
            output_states += (hidden_states,)
        if self.config.add_manual_pipeline_markers:
            # delaying the boundary here reduces the communciation memory
            mark_pipeline_boundary()

        return hidden_states, output_states


class FlaxCrossAttnUpBlock2D(nn.Module):
    r"""
    Cross Attention 2D Upsampling block - original architecture from Unet transformers:
    https://arxiv.org/abs/2103.06104
    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        dropout (:obj:`float`, *optional*, defaults to 0.0):
            Dropout rate
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        attn_num_head_channels (:obj:`int`, *optional*, defaults to 1):
            Number of attention heads of each spatial transformer block
        add_upsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add upsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    prev_output_channel: int
    config: UNet2DConfig
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []
        attentions = []

        for i in range(self.config.layers_per_block):
            res_skip_channels = self.in_channels if (
                i == self.config.layers_per_block - 1) else self.out_channels
            resnet_in_channels = self.prev_output_channel if i == 0 else self.out_channels

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                config=self.config,
                out_channels=self.out_channels,
                dtype=self.dtype,
            )
            resnets.append(res_block)

            attn_block = FlaxSpatialTransformer(
                in_channels=self.out_channels,
                n_heads=self.config.num_attention_heads,
                d_head=self.out_channels // self.config.num_attention_heads,
                depth=1,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

        self.resnets = resnets
        self.attentions = attentions

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels,
                                               dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 res_hidden_states_tuple,
                 temb,
                 encoder_hidden_states,
                 deterministic=True):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states),
                                            axis=-1)

            hidden_states = resnet(hidden_states,
                                   temb,
                                   deterministic=deterministic)
            hidden_states = attn(hidden_states,
                                 encoder_hidden_states,
                                 deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                mark_pipeline_boundary()

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)

        return hidden_states


class FlaxUpBlock2D(nn.Module):
    r"""
    Flax 2D upsampling block

    Parameters:
        in_channels (:obj:`int`):
            Input channels
        out_channels (:obj:`int`):
            Output channels
        prev_output_channel (:obj:`int`):
            Output channels from the previous block
        config (:obj:`UNet2DConfig`):
            UNet Global Config
        add_downsample (:obj:`bool`, *optional*, defaults to `True`):
            Whether to add downsampling layer before each final output
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    out_channels: int
    prev_output_channel: int
    config: UNet2DConfig
    add_upsample: bool = True
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        resnets = []

        for i in range(self.config.layers_per_block + 1):
            res_skip_channels = self.in_channels if (
                i == self.config.layers_per_block) else self.out_channels
            resnet_in_channels = (self.prev_output_channel
                                  if i == 0 else self.out_channels)

            res_block = FlaxResnetBlock2D(
                in_channels=resnet_in_channels + res_skip_channels,
                config=self.config,
                out_channels=self.out_channels,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets

        if self.add_upsample:
            self.upsamplers_0 = FlaxUpsample2D(self.out_channels,
                                               dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 res_hidden_states_tuple,
                 temb,
                 deterministic=True):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = jnp.concatenate((hidden_states, res_hidden_states),
                                            axis=-1)

            hidden_states = resnet(hidden_states,
                                   temb,
                                   deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                mark_pipeline_boundary()

        if self.add_upsample:
            hidden_states = self.upsamplers_0(hidden_states)
        return hidden_states


class FlaxUNetMidBlock2DCrossAttn(nn.Module):
    r"""
    Cross Attention 2D Mid-level block - original architecture from Unet transformers: https://arxiv.org/abs/2103.06104
    Parameters:
        in_channels (:obj:`int`):
            Input channels
        config (:obj:`UNet2DConfig`):
            UNet Global Config
        num_layers (:obj:`int`, *optional*, defaults to 1):
            Number of attention blocks layers
        dtype (:obj:`jnp.dtype`, *optional*, defaults to jnp.float32):
            Parameters `dtype`
    """
    in_channels: int
    config: UNet2DConfig
    num_layers: int = 1
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # there is always at least one resnet
        resnets = [
            FlaxResnetBlock2D(
                in_channels=self.in_channels,
                config=self.config,
                out_channels=self.in_channels,
                dtype=self.dtype,
            )
        ]

        attentions = []

        for _ in range(self.num_layers):
            attn_block = FlaxSpatialTransformer(
                in_channels=self.in_channels,
                n_heads=self.config.num_attention_heads,
                d_head=self.in_channels // self.config.num_attention_heads,
                depth=1,
                dtype=self.dtype,
            )
            attentions.append(attn_block)

            res_block = FlaxResnetBlock2D(
                in_channels=self.in_channels,
                config=self.config,
                out_channels=self.in_channels,
                dtype=self.dtype,
            )
            resnets.append(res_block)

        self.resnets = resnets
        self.attentions = attentions

    def __call__(self,
                 hidden_states,
                 temb,
                 encoder_hidden_states,
                 deterministic=True):
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states,
                                 encoder_hidden_states,
                                 deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                mark_pipeline_boundary()
            hidden_states = resnet(hidden_states,
                                   temb,
                                   deterministic=deterministic)
            if self.config.add_manual_pipeline_markers:
                mark_pipeline_boundary()

        return hidden_states


##### UNet2D
class FlaxUNet2DConditionModel(nn.Module):
    r"""
    FlaxUNet2DConditionModel is a conditional 2D UNet model that takes in a noisy sample, conditional state, and a
    timestep and returns sample shaped output.
    This model inherits from [`FlaxModelMixin`]. Check the superclass documentation for the generic methods the library
    implements for all the models (such as downloading or saving, etc.)
    Also, this model is a Flax Linen [flax.linen.Module](https://flax.readthedocs.io/en/latest/flax.linen.html#module)
    subclass. Use it as a regular Flax linen Module and refer to the Flax documentation for all matter related to
    general usage and behavior.
    Finally, this model supports inherent JAX features such as:
    - [Just-In-Time (JIT) compilation](https://jax.readthedocs.io/en/latest/jax.html#just-in-time-compilation-jit)
    - [Automatic Differentiation](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
    - [Vectorization](https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap)
    - [Parallelization](https://jax.readthedocs.io/en/latest/jax.html#parallelization-pmap)
    Parameters:
        config (:obj:`UNet2DConfig`):
            UNet Global Config
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use. The corresponding class names will be: "FlaxCrossAttnDownBlock2D",
            "FlaxCrossAttnDownBlock2D", "FlaxCrossAttnDownBlock2D", "FlaxDownBlock2D"
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D",)`):
            The tuple of upsample blocks to use. The corresponding class names will be: "FlaxUpBlock2D",
            "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D", "FlaxCrossAttnUpBlock2D"
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        cross_attention_dim (`int`, *optional*, defaults to 768):
            The dimension of the cross attention features.
    """

    config: UNet2DConfig
    down_block_types: Tuple[str] = (
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str] = ("UpBlock2D", "CrossAttnUpBlock2D",
                                  "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")
    block_out_channels: Tuple[int] = (320, 640, 1280, 1280)
    cross_attention_dim: int = 768
    dtype: jnp.dtype = jnp.float32

    def init_weights(self, rng: jax.random.PRNGKey) -> FrozenDict:
        # init input tensors
        sample_shape = (1, self.config.in_channels, self.config.sample_size,
                        self.config.sample_size)
        sample = jnp.zeros(sample_shape, dtype=jnp.float32)
        timesteps = jnp.ones((1,), dtype=jnp.int32)
        encoder_hidden_states = jnp.zeros((1, 1, self.cross_attention_dim),
                                          dtype=jnp.float32)

        params_rng, dropout_rng = jax.random.split(rng)
        rngs = {"params": params_rng, "dropout": dropout_rng}

        return self.init(rngs, sample, timesteps,
                         encoder_hidden_states)["params"]

    def setup(self):
        block_out_channels = self.block_out_channels
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = nn.Conv(
            block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

        # time
        self.time_proj = FlaxTimesteps(block_out_channels[0],
                                       freq_shift=self.config.freq_shift)
        self.time_embedding = FlaxTimestepEmbedding(time_embed_dim,
                                                    dtype=self.dtype)

        # down
        down_blocks = []
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(self.down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            if down_block_type == "CrossAttnDownBlock2D":
                down_block_cls = FlaxCrossAttnDownBlock2D
            else:
                down_block_cls = FlaxDownBlock2D
            down_block = down_block_cls(
                in_channels=input_channel,
                out_channels=output_channel,
                config=self.config,
                add_downsample=not is_final_block,
                dtype=self.dtype,
            )

            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        # mid
        self.mid_block = FlaxUNetMidBlock2DCrossAttn(
            in_channels=block_out_channels[-1],
            config=self.config,
            dtype=self.dtype,
        )

        # up
        up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(self.up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(
                i + 1,
                len(block_out_channels) - 1)]

            is_final_block = i == len(block_out_channels) - 1

            if up_block_type == "CrossAttnUpBlock2D":
                up_block_cls = FlaxCrossAttnUpBlock2D
            else:
                up_block_cls = FlaxUpBlock2D
            up_block = up_block_cls(
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                config=self.config,
                add_upsample=not is_final_block,
                dtype=self.dtype,
            )

            up_blocks.append(up_block)
            prev_output_channel = output_channel
        self.up_blocks = up_blocks

        # out
        self.conv_norm_out = nn.GroupNorm(num_groups=self.config.num_groups,
                                          epsilon=1e-5)
        self.conv_out = nn.Conv(
            self.config.out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            dtype=self.dtype,
        )

    def __call__(
        self,
        sample,
        timesteps,
        encoder_hidden_states,
        return_dict: bool = True,
        train: bool = False,
    ) -> Union[FlaxUNet2DConditionOutput, Tuple]:
        """r
        Args:
            sample (`jnp.ndarray`): (channel, height, width) noisy inputs tensor
            timestep (`jnp.ndarray` or `float` or `int`): timesteps
            encoder_hidden_states (`jnp.ndarray`): (channel, height, width) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] instead of a
                plain tuple.
            train (`bool`, *optional*, defaults to `False`):
                Use deterministic functions and disable dropout when not training.
        Returns:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition_flax.FlaxUNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`.
            When returning a tuple, the first element is the sample tensor.
        """
        # 1. time
        if not isinstance(timesteps, jnp.ndarray):
            timesteps = jnp.array([timesteps], dtype=jnp.int32)
        elif isinstance(timesteps, jnp.ndarray) and len(timesteps.shape) == 0:
            timesteps = timesteps.astype(dtype=jnp.float32)
            timesteps = jnp.expand_dims(timesteps, 0)

        t_emb = self.time_proj(timesteps)
        t_emb = self.time_embedding(t_emb)

        # 2. pre-process
        # (B, img_channel, sample_size, sample_size) -> (B, SS, SS, img_channel)
        sample = jnp.transpose(sample, (0, 2, 3, 1))
        # (B, SS, SS, block_out_channels[0])
        sample = self.conv_in(sample)
        if self.config.add_manual_pipeline_markers:
            mark_pipeline_boundary()

        # 3. down
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            if isinstance(down_block, FlaxCrossAttnDownBlock2D):
                sample, res_samples = down_block(sample,
                                                 t_emb,
                                                 encoder_hidden_states,
                                                 deterministic=not train)
            else:
                sample, res_samples = down_block(sample,
                                                 t_emb,
                                                 deterministic=not train)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample,
                                t_emb,
                                encoder_hidden_states,
                                deterministic=not train)

        # 5. up
        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-(
                self.config.layers_per_block + 1):]
            down_block_res_samples = down_block_res_samples[:-(
                self.config.layers_per_block + 1)]
            if isinstance(up_block, FlaxCrossAttnUpBlock2D):
                sample = up_block(
                    sample,
                    temb=t_emb,
                    encoder_hidden_states=encoder_hidden_states,
                    res_hidden_states_tuple=res_samples,
                    deterministic=not train,
                )
            else:
                sample = up_block(sample,
                                  temb=t_emb,
                                  res_hidden_states_tuple=res_samples,
                                  deterministic=not train)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = nn.silu(sample)
        sample = self.conv_out(sample)
        sample = jnp.transpose(sample, (0, 3, 1, 2))

        if not return_dict:
            return (sample,)

        return FlaxUNet2DConditionOutput(sample=sample)


def get_unet_2d(sample_size,
                down_block_types,
                up_block_types,
                block_out_channels,
                in_channels=4,
                out_channels=4,
                dropout=0.0,
                layers_per_block=2,
                num_attention_heads=8,
                freq_shift=0,
                num_groups=4,
                dtype=jnp.float32,
                add_manual_pipeline_markers=True):
    # Begin with Configs of Attention layers in the UNet_2D
    hidden_act = "gelu"
    hidden_size = block_out_channels[-1]
    # Check block out channels: only the last does not do upsampling
    assert block_out_channels[-1] == block_out_channels[-2]
    cross_attention_dim = block_out_channels[-1]
    config = UNet2DConfig(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=hidden_size * 4,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=dropout,
        hidden_act=hidden_act,
        add_manual_pipeline_markers=add_manual_pipeline_markers,
        # UNet New configs
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        freq_shift=freq_shift,
        num_groups=num_groups)
    return FlaxUNet2DConditionModel(config,
                                    down_block_types,
                                    up_block_types,
                                    block_out_channels,
                                    cross_attention_dim=cross_attention_dim,
                                    dtype=dtype)


if __name__ == "__main__":
    down_block_types: Tuple[str] = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
    )
    up_block_types: Tuple[str] = ("UpBlock2D", "UpBlock2D", "UpBlock2D",
                                  "UpBlock2D")
    block_out_channels: Tuple[int] = (32, 64, 128, 128)
    channel = 3
    sample_size = 24
    model = get_unet_2d(sample_size,
                        down_block_types,
                        up_block_types,
                        block_out_channels,
                        cross_attention_dim=128)
    rng = jax.random.PRNGKey(0)
    batch = 5
    sample = jnp.ones((batch, channel, sample_size, sample_size))
    encoder_hidden_states = jnp.ones(
        (batch, (sample_size // 2**(len(block_out_channels) - 1))**2,
         block_out_channels[-1]))
    timestep = 1
    params = model.init(rng, sample, timestep, encoder_hidden_states)
