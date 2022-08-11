import dataclasses
from dataclasses import dataclass
from functools import partial
import itertools
import math
import os
from typing import Callable, Optional, Tuple, Dict, Union

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
from configuration_utils import PretrainedConfig

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss

from ..utils import logging # Have to grab all logging into bloom_utils.py
# Have to copy and paste all of these classes
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

logger = logging.get_logger(__name__)




@flax.struct.dataclass
class BLOOMModelOutput(ModelOutput):
    last_hidden_state: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@flax.struct.dataclass
class BLOOMLMOutput(ModelOutput):
    logits: jax_xla.DeviceArray
    hidden_states: Optional[Tuple[jax_xla.DeviceArray]] = None
    attentions: Optional[Tuple[jax_xla.DeviceArray]] = None
    attention_cache: Optional[Tuple[Tuple[jax_xla.DeviceArray]]] = None


@dataclass(frozen=True)
class BloomConfig(PretrainedConfig):
#class BloomConfig:
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        vocab_size (`int`, *optional*, defaults to 50257):
            Vocabulary size of the Bloom model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`BloomModel`].
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the embeddings and hidden states.
        n_layer (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        n_head (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        attn_pdrop (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention.
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            The epsilon to use in the layer normalization layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            If enabled, use the layer norm of the hidden states as the residual in the transformer blocks
        skip_bias_add (`bool`, *optional*, defaults to `True`):
            If set to `True`, it will skip bias add for each linear layer in the transformer blocks
        skip_bias_add_qkv (`bool`, *optional*, defaults to `False`):
            If set to `True`, it will skip bias add for the first linear layer in the transformer blocks
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
            If set to `True` and the `dtype` is set to `float16` it will scale the input of the Softmax function to
            `fp32`
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate of the dropout function on the bias dropout.
        attention_dropout (`float`, *optional*, defaults to 0.1):
            Dropout rate applied to the attention probs
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Precision that has been used for the model's training in Megatron. Please load the model in the correct
            precision by doing `model = BloomModel.from_pretrained(model_name, torch_dtype="auto")`.`
        pretraining_tp (`int`, *optional*, defaults to `1`):
            Experimental feature. Tensor parallelism rank used during pretraining with Megatron. Please refer to [this
            document](https://huggingface.co/docs/transformers/parallelism) to understand more about it. This value is
            necessary to ensure exact reproducibility of the pretraining results. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). Note also that this is enabled only when
            `slow_but_exact=True`.
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            Experimental feature. Whether to use slow but exact implementation of the attention mechanism. While
            merging the TP rank tensors, due to slicing operations the results may be slightly different between the
            model trained on Megatron and our model. Please refer to [this
            issue](https://github.com/pytorch/pytorch/issues/76232). A solution to obtain more accurate results is to
            enable this feature. Enabling this will hurt the computational time of the inference. Will be probably
            resolved in the future once the main model has been fine-tuned with TP_rank=1.
    Example:
    ```python
    >>> from transformers import BloomModel, BloomConfig
    >>> # Initializing a Bloom configuration
    >>> configuration = BloomConfig()
    >>> # Initializing a model from the configuration
    >>> model = BloomModel(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "num_hidden_layers": "n_layer",
        "n_head": "num_attention_heads",
        "hidden_size": "n_embed",
        "dtype": "torch_dtype",
    }

    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        masked_softmax_fusion=True,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=False,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        attention_softmax_in_fp32=True,
        pretraining_tp=1,  # TP rank used when training with megatron
        dtype="bfloat16",
        slow_but_exact=False,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.masked_softmax_fusion = masked_softmax_fusion
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.dtype = dtype
        self.slow_but_exact = slow_but_exact

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.full((target_length, target_length), torch.finfo(dtype).min)
    mask_cond = torch.arange(mask.size(-1))
    intermediate_mask = mask_cond < (mask_cond + 1).view(mask.size(-1), 1)
    mask.masked_fill_(intermediate_mask, 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(target_length, past_key_values_length, dtype=dtype), mask], dim=-1)
    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: int = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    batch_size, source_length = mask.size()
    tgt_len = tgt_len if tgt_len is not None else source_length

    expanded_mask = mask[:, None, None, :].expand(batch_size, 1, tgt_len, source_length).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def build_alibi_tensor(attention_mask: torch.Tensor, n_head: int, dtype, device) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    Args:
    Returns tensor shaped (batch_size * n_head, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        n_head (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
        device (`torch.device`, *optional*, default=`torch.device('cpu')`):
            device of the output alibi tensor
    """
    closest_power_of_2 = 2 ** math.floor(math.log2(n_head))
    base = torch.tensor(2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=device, dtype=torch.float32)
    powers = torch.arange(1, 1 + closest_power_of_2, device=device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != n_head:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, n_head - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=n_head, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    # batch_size = 1, n_head = n_head, query_length

    arange_tensor = (attention_mask.cumsum(-1)[:, None, :].to(device) - 1) * attention_mask[:, None]
    alibi = slopes.unsqueeze(-1) * arange_tensor
    alibi = alibi * attention_mask[:, None]
    return alibi.reshape(alibi.shape[0] * n_head, 1, -1).to(dtype)


def dropout_add(x, residual, prob, training):
    """
    Dropout add function
    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *rquired*):
            esidual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = nn.functional.dropout(x, p=prob, training=training)
    out = residual + out
    return out


def bloom_gelu_forward(x):
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.
    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


def bloom_gelu_back(g, x):
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)
    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


class BloomGelu(nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs
    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


class BloomAttention(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.split_size = self.hidden_size
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        # Layer-wise attention scaling
        self.layer_number = max(1, layer_number)
        self.norm_factor = math.sqrt(self.head_dim) * self.layer_number

        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, fused_qkv):
        """
        Split the last dimension into (num_heads, head_dim)
        """
        new_tensor_shape = fused_qkv.size()[:-1] + (self.num_heads, 3 * self.head_dim)
        # new_tensor_shape = (fused_qkv.size(1), fused_qkv.size(0)*fused_qkv.size(2), fused_qkv.size(-1))
        # fused_qkv = fused_qkv.transpose(1, 0)
        fused_qkv = fused_qkv.reshape(*new_tensor_shape)
        # fused_qkv = fused_qkv.permute(0, 2, 1, 3)
        return torch.split(fused_qkv, self.head_dim, -1)

    def _merge_heads(self, x):
        # What we want to achieve is:
        # batch_size * num_heads, seq_len, head_dim -> batch_size, seq_len, num_heads * head_dim

        # First view to decompose the batch size
        # batch_size*num_heads, seq_len, head_dim -> batch_size, num_heads, seq_len, head_dim
        x = x.view(x.size(0) // self.num_heads, self.num_heads, x.size(1), self.head_dim)

        # batch_size, num_heads, seq_len, head_dim -> batch_size, seq_len, num_heads, head_dim
        x = x.permute(0, 2, 1, 3)

        # batch_size, seq_len, num_heads, head_dim -> batch_size, seq_len, num_heads * head_dim
        return x.reshape(x.size(0), x.size(1), self.num_heads * self.head_dim)

    def forward(
        self,
        hidden_states,
        residual,
        layer_past=None,
        attention_mask=None,
        alibi=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
    ):
        alibi = alibi.to(hidden_states.device)  # to make the model possible to run under accelerate
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = self._split_heads(fused_qkv)

        if layer_past is not None:
            past_key, past_value = layer_past
            # concatenate along seq_length dimension -> [batch_size, qk_length, num_heads, head_dim]
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=1)
            value_layer = torch.cat((past_value.type_as(value_layer), value_layer), dim=1)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        beta = 1.0 / self.layer_number

        # # [batch_size*num_heads, head_dim, q_length] x [batch_size*num_heads, head_dim, k_length] -> [batch_size*num_heads, q_length, k_length]
        matmul_result = (1.0 / self.norm_factor) * torch.bmm(
            query_layer.transpose(1, 2).reshape(-1, query_layer.shape[1], query_layer.shape[3]),
            key_layer.permute(0, 2, 3, 1).reshape(-1, key_layer.shape[3], key_layer.shape[1]),
        ) + beta * alibi

        # change view to [batch_size, num_heads, q_length, k_length]
        attention_scores = matmul_result.view(-1, self.num_heads, matmul_result.size(1), matmul_result.size(2))

        # We replace the scaled softmax by just a few line of code - [batch_size, num_heads, q_length, k_length]
        input_dtype = attention_scores.dtype
        attn_weights = (attention_scores * self.layer_number) + attention_mask
        attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
        attention_probs = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)
        attention_probs = attention_probs * (~attention_mask.bool())
        # [batch_size, num_heads, q_length, k_length]
        attention_probs = self.attention_dropout(attention_probs)

        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # change view [batch_size x num_heads, q_length, k_length]
        attention_probs_reshaped = attention_probs.view(*matmul_result.shape)

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = torch.bmm(
            attention_probs_reshaped, value_layer.transpose(1, 2).reshape(-1, value_layer.size(1), value_layer.size(3))
        )

        # change view [batch_size, num_heads, q_length, head_dim]
        context_layer = self._merge_heads(context_layer)

        # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
        if self.pretraining_tp > 1 and self.slow_but_exact:
            slices = context_layer.shape[-1] / self.pretraining_tp
            output_tensor = torch.zeros_like(context_layer)
            for i in range(self.pretraining_tp):
                output_tensor = output_tensor + nn.functional.linear(
                    context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            output_tensor = self.dense(context_layer)

        output_tensor = dropout_add(output_tensor, residual, self.hidden_dropout, self.training)

        outputs = (output_tensor, present)
        if output_attentions:
            outputs += (attention_probs,)

        return outputs


class BloomMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = nn.Linear(hidden_size, 4 * hidden_size)
        self.dense_4h_to_h = nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout
        self.gelu_impl = BloomGelu()

    def forward(self, hidden_states, residual):
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + nn.functional.linear(
                    hidden_states[:, :, int(i * slices) : int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices) : int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = dropout_add(intermediate_output, residual, self.hidden_dropout, self.training)

        return output


class BloomBlock(nn.Module):
    def __init__(self, config, layer_number=None):
        super().__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.n_head = config.n_head
        self.self_attention = BloomAttention(config, layer_number=layer_number)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
        output_attentions=False,
        alibi=None,
    ):
        # hidden_states: [batch_size, seq_length, hidden_size]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)

        # Layer norm post the self attention.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states

        # Self attention.
        attn_outputs = self.self_attention(
            layernorm_output,
            residual,
            layer_past=layer_past,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        attention_output = attn_outputs[0]

        outputs = attn_outputs[1:]

        layernorm_output = self.post_attention_layernorm(attention_output)

        # Get residual
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = attention_output

        # MLP.
        output = self.mlp(layernorm_output, residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (output,) + outputs[1:]

        return outputs  # hidden_states, present, attentions


class BloomPreTrainedModel(PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BloomConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BloomBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BloomModel):
            module.gradient_checkpointing = value


BLOOM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`BloomConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

BLOOM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.
            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.
            Indices can be obtained using [`BloomTokenizerFast`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[Tuple[torch.Tensor]]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.
            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
"""
class BloomModel(BloomPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size
        self.n_head = config.n_head

        # Embedding + LN Embedding
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)

        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Transformer blocks
        self.h = nn.ModuleList([BloomBlock(config, layer_number=i) for i in range(config.num_hidden_layers)])

        # Final Layer Norm
        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(attention_mask.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings):
        self.word_embeddings = new_embeddings

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if past_key_values is None:
            past_key_values = tuple([None] * len(self.h))

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_head x N x N
        # head_mask has shape n_layer x batch x n_head x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        current_sequence_length = hidden_states.shape[1]
        past_key_values_length = 0
        if past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[1]
            current_sequence_length += past_key_values_length

        if attention_mask is None:
            attention_mask = torch.ones((hidden_states.shape[0], current_sequence_length), device=hidden_states.device)
        else:
            attention_mask = attention_mask.to(hidden_states.device)

        alibi = build_alibi_tensor(attention_mask, self.n_head, hidden_states.dtype, hidden_states.device)

        causal_mask = self._prepare_attn_mask(attention_mask, input_shape, inputs_embeds, past_key_values_length)

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions, alibi)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    causal_mask,
                    head_mask[i],
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=causal_mask,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    alibi=alibi,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = hidden_states.view(output_shape)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


@add_start_docstrings(
    """
    The Bloom Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = BloomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(past: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past
        )


@add_start_docstrings(
    """
    The Bloom Model transformer with a sequence classification head on top (linear layer).
    [`BloomForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-1) do.
    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForSequenceClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = BloomModel(config)
        self.score = nn.Linear(config.hidden_size, config.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    Bloom Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for
    Named-Entity-Recognition (NER) tasks.
    """,
    BLOOM_START_DOCSTRING,
)
class BloomForTokenClassification(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = BloomModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class BloomEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BloomConfig
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


class BloomSelfAttention(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        if self.config.decoder_embed_dim % self.config.decoder_attention_heads != 0:
            raise ValueError(
                f"`decoder_embed_dim`: {self.config.decoder_embed_dim} has to be a "
                f"multiple of `decoder_attention_heads`: {self.config.decoder_attention_heads}"
            )

        self.qvk_combined = nn.Dense(
            self.config.decoder_embed_dim * 3,
            dtype=self.dtype,
        )

    def __call__(self,
                 hidden_states,
                 output_attentions: bool = False,
                 attention_cache=None):
        head_dim = self.config.decoder_embed_dim // self.config.decoder_attention_heads

        qvk_combined_states = self.qvk_combined(hidden_states)
        qvk_combined_states = qvk_combined_states.reshape(
            qvk_combined_states.shape[:2] + (-1, 3))
        query_states, value_states, key_states = jnp.split(qvk_combined_states,
                                                           3,
                                                           axis=3)

        query_states = query_states.reshape(hidden_states.shape[:2] + (
            self.config.decoder_attention_heads, head_dim))
        value_states = value_states.reshape(hidden_states.shape[:2] + (
            self.config.decoder_attention_heads, head_dim))
        key_states = key_states.reshape(hidden_states.shape[:2] +
                                        (self.config.decoder_attention_heads,
                                         head_dim))

        if attention_cache is None:
            attention_bias = jnp.expand_dims(
                jnp.triu(
                    jnp.full((query_states.shape[1], key_states.shape[1]),
                             -1e10), 1), (0, 1))
        else:
            cache_key, cache_value, cache_index = attention_cache
            cache_index_ = cache_index[0]
            key_states = lax.dynamic_update_slice(cache_key, key_states,
                                                  (0, cache_index_, 0, 0))
            value_states = lax.dynamic_update_slice(cache_value, value_states,
                                                    (0, cache_index_, 0, 0))
            num_updated_cache_vectors = query_states.shape[1]
            max_length = key_states.shape[1]

            # The following logic is equivalent to:
            # attention_bias = jnp.expand_dims(
            #     jnp.triu(jnp.full(
            #         (num_updated_cache_vectors, max_length), -1e10), cache_index + 1), (0, 1))
            # but "cache_index + 1" in jnp.triu results in non-static IR.
            row_idxs = jnp.arange(num_updated_cache_vectors)
            mask = jnp.arange(max_length) - (cache_index_ + 1)
            attention_bias = jnp.expand_dims(
                (row_idxs[:, None] <= mask).astype(self.dtype) * -1e10, (0, 1))

            attention_cache = key_states, value_states, cache_index + num_updated_cache_vectors
        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
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


# class OPTAttention(nn.Module):
#     config: OPTConfig
#     dtype: jnp.dtype = jnp.float16

#     def setup(self):
#         assert self.config.decoder_normalize_before
#         self.self = OPTSelfAttention(self.config, dtype=self.dtype)
#         self.dense = nn.Dense(
#             self.config.decoder_embed_dim,
#             dtype=self.dtype,
#         )
#         self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
#                                        dtype=self.dtype)

#     def __call__(self,
#                  hidden_states,
#                  output_attentions: bool = False,
#                  attention_cache=None):
#         residual = hidden_states
#         hidden_states = self.layer_norm(hidden_states)
#         attn_outputs = self.self(hidden_states,
#                                  output_attentions=output_attentions,
#                                  attention_cache=attention_cache)
#         attn_output = attn_outputs[0]
#         attention_cache = attn_outputs[1]
#         hidden_states = self.dense(attn_output)
#         hidden_states = hidden_states + residual
#         outputs = (hidden_states, attention_cache)

#         if output_attentions:
#             outputs += (attn_outputs[2],)

#         return outputs


class BloomFFN(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_embed_dim,
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.activation_fn]
        self.fc2 = nn.Dense(
            self.config.decoder_embed_dim,
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


class BloomTransformerLayer(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert self.config.decoder_normalize_before
        assert not getattr(self.config, "cross_self_attention", False)
        assert not getattr(self.config, "scale_heads", False)
        assert not getattr(self.config, "scale_attn", False)
        assert not getattr(self.config, "scale_fc", False)
        self.attention = BloomAttention(self.config, dtype=self.dtype)
        self.ffn = BloomFFN(self.config, dtype=self.dtype)

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


class BloomTransformerLayerCollection(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        self.layers = [
            BloomTransformerLayer(self.config, name=str(i), dtype=self.dtype)
            for i in range(self.config.decoder_layers)
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

        return BloomModelOutput(last_hidden_state=hidden_states,
                              hidden_states=all_hidden_states,
                              attentions=all_attentions,
                              attention_cache=new_attention_cache)


class BloomTransformerModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16  # the dtype of the computation

    def setup(self):
        assert self.config.decoder_normalize_before
        self.embeddings = BloomEmbeddings(self.config, dtype=self.dtype)
        self.encoder = BloomTransformerLayerCollection(self.config,
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

        return BloomModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )


class BloomForLMModule(nn.Module):
    config: BloomConfig
    dtype: jnp.dtype = jnp.float16
    bias_init: Callable[..., jnp.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = BloomTransformerModule(config=self.config,
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

        return BloomLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            attention_cache=outputs.attention_cache,
        )


def get_bloom_config(name, **kwargs):
    if name == "125M":
        config = BloomConfig(
            max_target_positions=2048,
            decoder_layers=12,
            decoder_attention_heads=12,
            decoder_embed_dim=768,
            decoder_input_dim=768,
            decoder_ffn_embed_dim=768 * 4,
            version=3,
        )
    elif name == "2.7B":
        config = BloomConfig(
            max_target_positions=2048,
            decoder_layers=32,
            decoder_attention_heads=32,
            decoder_embed_dim=2560,
            decoder_input_dim=2560,
            decoder_ffn_embed_dim=2560 * 4,
            version=3,
        )
    elif name == "6.7B":
        config = BloomConfig(
            max_target_positions=2048,
            decoder_layers=32,
            decoder_attention_heads=32,
            decoder_embed_dim=4096,
            decoder_input_dim=4096,
            decoder_ffn_embed_dim=4096 * 4,
            version=3,
        )
    elif name == "30B":
        config = BloomConfig(
            max_target_positions=2048,
            decoder_layers=48,
            decoder_attention_heads=56,
            decoder_embed_dim=7168,
            decoder_input_dim=7168,
            decoder_ffn_embed_dim=7168 * 4,
            version=3,
        )
    elif name == "175B":
        config = BloomConfig(
            max_target_positions=2048,
            decoder_layers=96,
            decoder_attention_heads=96,
            decoder_embed_dim=12288,
            decoder_input_dim=12288,
            decoder_ffn_embed_dim=12288 * 4,
            version=3,
        )
    else:
        raise ValueError()

    return dataclasses.replace(config, **kwargs)


def init_model_aval(config):
    """Initialize model with parameters with abstract values (shape-only arrays)."""
    model = BloomForLMModule(config, dtype=config.dtype)
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
    for i in range(config.decoder_layers):
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


def load_params_np(params, path, config, dummy=False):
    """Load parameterswith numpy arrays."""
    if dummy:
        np_dtype = np.float32 if config.dtype == jnp.float32 else np.float16
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
    for i in tqdm(range(config.decoder_layers)):
        param_prefix = f"params.transformers.encoder.{i}."
        load_prefix = f"decoder.layers.{i}."
        # Attention weights
        wq = load_array(load_prefix + "self_attn.q_proj.weight")
        wk = load_array(load_prefix + "self_attn.k_proj.weight")
        wv = load_array(load_prefix + "self_attn.v_proj.weight")
        dim = wq.shape[-1]
        w_qvk = np.concatenate([wq, wv, wk], axis=0).reshape(
            (3, -1, dim)).transpose([2, 1, 0]).reshape((dim, -1))
        load_param(param_prefix + "attention.self.qvk_combined.kernel", w_qvk)
        bq = load_array(load_prefix + "self_attn.q_proj.bias")
        bk = load_array(load_prefix + "self_attn.k_proj.bias")
        bv = load_array(load_prefix + "self_attn.v_proj.bias")
        b_qvk = np.concatenate([bq, bv, bk], axis=0).reshape(
            (3, dim)).transpose([1, 0]).reshape((-1,))
        load_param(param_prefix + "attention.self.qvk_combined.bias", b_qvk)
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


def get_jax_executable(config,
                       support_output_attentions=False,
                       support_output_hidden_states=False):
    model, params = init_model_aval(config)

    @jax.jit
    def inference_step(params, batch):
        output = model.apply(params,
                             batch["input_ids"],
                             batch["position_ids"],
                             attention_cache=batch["cache"],
                             output_attentions=support_output_attentions,
                             output_hidden_states=support_output_hidden_states)
        return output

    executable = inference_step
    return executable, params


def get_pipeshard_executable(config,
                             batch_size=1,
                             num_micro_batches=1,
                             decoding_length_per_step=1024,
                             support_output_attentions=False,
                             support_output_hidden_states=False,
                             autoregressive=True):

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

        @alpa.parallelize(batch_argnums=(1,), method=method)
        def inference_step_with_cache(params, batch):
            output = model.apply(
                params,
                batch["input_ids"],
                batch["position_ids"],
                attention_cache=batch["cache"],
                output_attentions=support_output_attentions,
                output_hidden_states=support_output_hidden_states)
            return output

        alpa.global_config.always_donate_micro_batch_vars = False
        executable = inference_step_with_cache.get_executable(
            params, {
                "input_ids":
                    jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "position_ids":
                    jax.core.ShapedArray((batch_size, 1), jnp.int32),
                "cache":
                    init_cache_aval(config, batch_size),
            })
    else:

        @alpa.parallelize(batch_argnums=(1,), method=method)
        def inference_step(params, batch):
            output = model.apply(
                params,
                batch["input_ids"],
                batch["position_ids"],
                output_attentions=support_output_attentions,
                output_hidden_states=support_output_hidden_states)
            return output

        assert batch_size % num_micro_batches == 0, "cannot divide batch_size by num_micro_batches"
        micro_batch_size = batch_size // num_micro_batches

        executable = inference_step.get_executable(
            params, {
                "input_ids":
                    jax.core.ShapedArray(
                        (batch_size, decoding_length_per_step), jnp.int32),
                "position_ids":
                    jax.core.ShapedArray(
                        (batch_size, decoding_length_per_step), jnp.int32),
            })

    executable.dump_debug_info("tmp")
    return executable, params


def load_bloom_params_worker_func(self, path, prefix_to_idx, config, shapes,
                                uuids, indices, mesh_ids):
    """The worker function to load Bloom parameters."""

    def load_array(key):
        return np.load(os.path.join(path, key))

    def load_param(param_key, loaded_array):
        i = prefix_to_idx[param_key]

        for j in range(len(mesh_ids[i])):
            if self.mesh_id != mesh_ids[i][j]:
                continue

            assert shapes[i][j] == loaded_array.shape
            uuid = uuids[i][j]
            datas = []
            for k in range(len(self.local_devices)):
                idx = self.host_id * len(self.local_devices) + k
                datas.append(loaded_array[indices[i][j][idx]])
            self.put_buffers(uuid, datas)

    load_param("params.transformers.embeddings.word_embeddings.embedding",
               load_array("decoder.embed_tokens.weight"))
    load_param("params.transformers.embeddings.position_embeddings.embedding",
               load_array("decoder.embed_positions.weight"))

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
        w_qvk = np.concatenate([wq, wv, wk], axis=0).reshape(
            (3, -1, dim)).transpose([2, 1, 0]).reshape((dim, -1))
        load_param(param_prefix + "attention.self.qvk_combined.kernel", w_qvk)
        bq = load_array(load_prefix + "self_attn.q_proj.bias")
        bk = load_array(load_prefix + "self_attn.k_proj.bias")
        bv = load_array(load_prefix + "self_attn.v_proj.bias")
        b_qvk = np.concatenate([bq, bv, bk], axis=0).reshape(
            (3, dim)).transpose([1, 0]).reshape((-1,))
        load_param(param_prefix + "attention.self.qvk_combined.bias", b_qvk)
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
            w.load_bloom_params_worker_func.remote(path, prefix_to_flat_idx,
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
