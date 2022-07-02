import unittest
from enum import Enum
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor, embedding
import alpa.torch.optim as torchoptim
from alpa.torch.trainer import train_torch_module
import alpa


# Copied from timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "gelu":
        return torch.nn.functional.gelu

    raise RuntimeError(
        "activation should be relu/gelu, not {}".format(activation))


# Adapted from torch/nn/modules/transformer.py
class TransformerEncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = "relu",
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = Attention(d_model, num_heads=nhead, attn_drop=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask,
                                   src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x +
                           self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:
        # x = self.self_attn(x, x, x,
        #                    attn_mask=attn_mask,
        #                    key_padding_mask=key_padding_mask,
        #                    need_weights=False)[0]
        # TODO: add support for `attn_mask` / `key_padding_mask` if needed.
        x = self.self_attn(x)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TokenMixer(Enum):
    DOT = 1
    LINEAR = 2
    ATTENTION = 3
    CONVOLUTION = 4


# util for generating a weight and a bias based on a size, and initializing them
def construct_w_b_pair(
        shape: List[int],
        uniform_const: float) -> Tuple[nn.Parameter, nn.Parameter]:
    assert len(shape) == 2
    w = nn.Parameter(
        torch.empty(shape).uniform_(-1 * uniform_const, uniform_const))
    b = nn.Parameter(
        torch.empty([shape[0]]).uniform_(-1 * uniform_const,
                                         uniform_const))  # UniformFillß

    return w, b


# The implementation of ZHEN layer is based on the paper: https://arxiv.org/pdf/2203.11014.pdf
#
# This is a single ZHEN layer. It:
# - receives an input from the previous layer, or the embedding (first layer)
# - receives the skip connection, which is the input to the previous layer (or nothing, in the case of first ZHEN layer)
# - adds input and skip connection together, and treat it as the new input
# and runs the new input through the different modules in token_mixer_list one by one, and concat them together as the ensemble.
# It outputs the ensemble result and the new input
# see https://bit.ly/3wNuqfz for a visualization.
class ZHENLayer(nn.Module):

    def __init__(
        self,
        layer_index: int,
        emb_dim: int,
        token_mixer_list: List[
            TokenMixer],  # determines this layer's output features
        previous_n_embs:
        int = 369,  # previous layer's output dim, may not be inferrable if token_mixer is different per layer. If 0th layer, this is original_n_embs.
        previous_input_embs:
        int = 369,  # skip connection's num embs. This is previous layer's input num embs.
        output_embs_per_mixer: int = 50,  # each module outputs 50 embeddings
        original_n_embs:
        int = 369,  # whatever overarch gives us for the 0th zhen layer . the rest, is whatever output previous layer is
    ):
        super().__init__()
        self.layer_index = layer_index
        self.emb_dim = emb_dim
        self.token_mixer_list = token_mixer_list
        self.mismatched_skip_and_input_shape = previous_n_embs != previous_input_embs
        if token_mixer_list is not None:
            self.token_mixer_list = token_mixer_list
        # self.sum_for_skip = sum_for_skip
        zhen_n_embs = len(token_mixer_list) * output_embs_per_mixer
        self.n_embs = zhen_n_embs
        if self.layer_index != 0:
            if self.mismatched_skip_and_input_shape:
                self.match_w, self.match_b = construct_w_b_pair(
                    [previous_n_embs, previous_input_embs], 0.0)

        self.layer_norm_w = nn.Parameter(torch.empty(
            [emb_dim]).fill_(0.0))  # ConstantFill
        self.layer_norm_b = nn.Parameter(torch.empty(
            [emb_dim]).fill_(0.0))  # ConstantFill
        for token_mixer in self.token_mixer_list:
            if token_mixer == TokenMixer.DOT:
                self.ffn_w, self.ffn_b = construct_w_b_pair(
                    [
                        512,
                        original_n_embs**2
                        if self.layer_index == 0 else previous_n_embs**2,
                    ],
                    0.03125,
                )
                self.pool_w, self.pool_b = construct_w_b_pair(
                    [
                        output_embs_per_mixer * emb_dim,
                        512,
                    ],
                    0.3125,
                )
            elif token_mixer == TokenMixer.LINEAR:  # n = 50
                self.w_linear, self.b_linear = construct_w_b_pair(
                    [output_embs_per_mixer, previous_n_embs], 0.0)

            elif token_mixer == TokenMixer.ATTENTION:  # n = 50
                self.encoder_layer = TransformerEncoderLayer(d_model=emb_dim,
                                                             nhead=1,
                                                             batch_first=True)

                self.w_attention, self.b_attention = construct_w_b_pair(
                    [output_embs_per_mixer, previous_n_embs], 0.0)

            elif token_mixer == TokenMixer.CONVOLUTION:
                self.conv = nn.Conv2d(1, 1, 5, stride=1, padding=(2, 2))
                self.w_conv, self.b_conv = construct_w_b_pair(
                    [
                        output_embs_per_mixer,
                        original_n_embs
                        if self.layer_index == 0 else previous_n_embs,
                    ],
                    0.0,
                )

    def get_dense_params(self) -> List[nn.Parameter]:
        # do not save because this may turn into FSDP
        return list(self.parameters())

    def forward(
            self,
            skip_connection: Optional[
                torch.
                Tensor],  # the skip connection, i.e., previous layer's input
            input: torch.Tensor,  # this is previous layer's ensemble output
            # B, D, F
    ):
        B = input.shape[0]
        # process orig embs
        # token mixer not None
        if self.layer_index != 0:
            if self.mismatched_skip_and_input_shape:
                skip_connection = torch.nn.functional.linear(skip_connection,
                                                             self.match_w,
                                                             bias=self.match_b)
            input_feature = skip_connection + input
        else:
            # 0th layer, no skip
            input_feature = input

        output = []  # do not call cat N times. Call it once.
        for token_mixer in self.token_mixer_list:
            if token_mixer == TokenMixer.DOT:  # num_dot_emb = 50
                # B,D,F
                input_feature_t = input_feature.permute(0, 2, 1)
                # B,F,D
                dot_products = torch.bmm(input_feature_t, input_feature)
                # B,F,F
                flattened_dot_products = torch.flatten(dot_products,
                                                       start_dim=-2)  # Flatten
                # B,F**2
                r = torch.addmm(self.ffn_b, flattened_dot_products,
                                self.ffn_w.t())  # FC
                r_act = torch.relu(r)  # Relu
                r_pooled = torch.nn.functional.linear(
                    r_act,
                    self.pool_w,
                    bias=self.pool_b,
                )
                output.append(r_pooled.view(B, -1, self.emb_dim))

            elif token_mixer == TokenMixer.LINEAR:
                linear_emb_list = torch.nn.functional.linear(input_feature,
                                                             self.w_linear,
                                                             bias=self.b_linear)
                flat_linear_emb_list = linear_emb_list.permute(0, 2, 1)
                output.append(flat_linear_emb_list)

            elif token_mixer == TokenMixer.ATTENTION:
                # input: B,D,F
                compress_list = torch.nn.functional.linear(
                    input_feature, self.w_attention, bias=self.b_attention)
                # B,D,O
                compress_list_t = compress_list.permute(0, 2, 1)  # (B,O,D)
                attention_emb_list = self.encoder_layer(compress_list_t)
                output.append(attention_emb_list)

            elif token_mixer == TokenMixer.CONVOLUTION:
                reshape_input_feature = input_feature.reshape(
                    B, 1, self.emb_dim, -1)
                r_conv = self.conv(reshape_input_feature)
                reshape_r_conv = r_conv.reshape(B, self.emb_dim, -1)
                compress_list = torch.nn.functional.linear(
                    reshape_r_conv, self.w_conv, bias=self.b_conv)  # B,output,D
                flat_compress_list = compress_list.permute(0, 2, 1)
                output.append(flat_compress_list)
            else:
                assert 0, f"unknown module: {token_mixer}"

        # each output should be B,F,D
        output = torch.cat(output, dim=1)
        output_embs = torch.nn.functional.layer_norm(
            output,
            output.size()[2:],
            weight=self.layer_norm_w,
            bias=self.layer_norm_b,
        )
        return output_embs.permute(0, 2, 1), input_feature


# ZHEN collection is different ZHEN layers
class ZHENCollection(nn.Module):

    def __init__(
        self,
        num_layers: int,
        emb_dim: int,
        token_mixer_list: Union[List[TokenMixer], List[List[TokenMixer]]],
        original_emb_num: int,
        output_emb_per_ensemble_module: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.token_mixer_list = token_mixer_list
        self.layers: nn.ModuleList = nn.ModuleList([])

        assert len(token_mixer_list) > 0
        if type(token_mixer_list[0]) == list:
            # this is a heterogeneous ZHEN
            assert num_layers == len(
                token_mixer_list
            ), "if token_mixer_list is a list of list of modules, ensure num_layers = len(token_mixer_list)"  # noqa
        else:
            # this is a homogeneous ZHEN. Convert it to heterogeneous ZHEN
            # pyre-ignore
            token_mixer_list = [token_mixer_list] * num_layers

        for i in range(num_layers):
            layer = ZHENLayer(
                layer_index=i,
                emb_dim=emb_dim,
                # pyre-ignore[6]
                token_mixer_list=token_mixer_list[i],
                previous_n_embs=(
                    original_emb_num if i == 0
                    # pyre-ignore[6]
                    else len(token_mixer_list[i - 1]) *
                    output_emb_per_ensemble_module),
                previous_input_embs=(
                    original_emb_num if i <= 1
                    # pyre-ignore[6]
                    else len(token_mixer_list[i - 2]) *
                    output_emb_per_ensemble_module),
                output_embs_per_mixer=output_emb_per_ensemble_module,
                original_n_embs=original_emb_num,
            )
            self.layers.append(layer)

    def forward(
        self,
        input: torch.Tensor,
        skip_connection: Optional[torch.Tensor] = None,
    ):
        skip_connection = None  # previous layer's input
        for layer in self.layers:
            input, skip_connection = layer(skip_connection, input)

        output = input.reshape(input.shape[0], -1)
        return output

    def get_dense_params(self) -> List[nn.Parameter]:
        return list(self.parameters())


def weight_init_func(pt_module, name_map, params, bufs):
    # for k, m in pt_module.named_modules():
    #     if isinstance(m, torch.nn.Linear):
    #         params[name_map[f"{k}.weight"]] = torch.nn.init.xavier_uniform(params[name_map[f"{k}.weight"]])
    #         params[name_map[f"{k}.bias"]] = torch.nn.init.normal(params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


class TorchZHENTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)
        alpa.set_seed(123)

    def test_zhen_homogeneous(self):
        B = 64  # made multiples of 8
        F = 48  # made multiples of 8
        D = 64
        LAYERS = 5
        OUTPUT_PER_ENSEMBLE = 48  # made multiples of 8
        TOKENS = [
            TokenMixer.ATTENTION, TokenMixer.LINEAR, TokenMixer.ATTENTION,
            TokenMixer.CONVOLUTION, TokenMixer.DOT
        ]

        pt_module_gen = lambda: ZHENCollection(LAYERS, D, TOKENS, F,
                                               OUTPUT_PER_ENSEMBLE)

        dataloader = [(torch.empty(
            B, D, F), torch.empty(B, D * LAYERS * OUTPUT_PER_ENSEMBLE))] * 2
        loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
            *args, **kwargs)
        optim_gen = torchoptim.adam(lr=1e-3)
        num_micro_batches = 2
        parallel_method = alpa.PipeshardParallel(
            num_micro_batches=num_micro_batches,
            layer_option=alpa.AutoLayerOption(layer_num=2),
            stage_option="auto")

        _xla_client_mem_fraction_orig_value = alpa.global_config.xla_client_mem_fraction
        alpa.global_config.xla_client_mem_fraction = 0.7

        train_torch_module(pt_module_gen, weight_init_func, dataloader,
                           loss_func, optim_gen, parallel_method)

        alpa.global_config.xla_client_mem_fraction = _xla_client_mem_fraction_orig_value

    def test_zhen_heterogeneous(self):
        B = 64
        F = 37
        D = 64
        OUTPUT_PER_ENSEMBLE = 48  # 50  # made multiples of 8
        TOKENS = [[TokenMixer.ATTENTION, TokenMixer.LINEAR],
                  [
                      TokenMixer.ATTENTION, TokenMixer.CONVOLUTION,
                      TokenMixer.DOT
                  ], [TokenMixer.LINEAR, TokenMixer.DOT]]  # 3-layer ZHEN

        pt_module_gen = lambda: ZHENCollection(len(TOKENS), D, TOKENS, F,
                                               OUTPUT_PER_ENSEMBLE)

        dataloader = [(torch.empty(
            B, D, F), torch.empty(B,
                                  D * len(TOKENS[-1]) * OUTPUT_PER_ENSEMBLE))
                     ] * 2
        loss_func = lambda *args, **kwargs: torch.nn.functional.mse_loss(
            *args, **kwargs)
        optim_gen = torchoptim.adam(lr=1e-3)
        num_micro_batches = 2
        parallel_method = alpa.PipeshardParallel(
            num_micro_batches=num_micro_batches,
            layer_option=alpa.AutoLayerOption(layer_num=2),
            stage_option="auto")

        _xla_client_mem_fraction_orig_value = alpa.global_config.xla_client_mem_fraction
        alpa.global_config.xla_client_mem_fraction = 0.7

        train_torch_module(pt_module_gen, weight_init_func, dataloader,
                           loss_func, optim_gen, parallel_method)

        alpa.global_config.xla_client_mem_fraction = _xla_client_mem_fraction_orig_value


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TorchZHENTest("test_zhen_homogeneous"))
    suite.addTest(TorchZHENTest("test_zhen_heterogeneous"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
