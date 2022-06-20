import unittest

import torch
from torch import nn
import alpa.torch.optim as torchoptim
import alpa
from alpa.torch.trainer import train_torch_module
import numpy as np
# from pycls.core.config import cfg
# from pycls.models.blocks import (
#     # MultiheadAttention,
#     activation,
#     conv2d,
#     layernorm,
#     linear,
#     norm2d,
#     patchify2d,
# )


def activation(activation_fun=None):
    """Helper for building an activation layer."""
    activation_fun = (activation_fun or "relu").lower()
    if activation_fun == "relu":
        return nn.ReLU(inplace=False)
    elif activation_fun == "silu" or activation_fun == "swish":
        try:
            return nn.SiLU()
        except AttributeError:
            return SiLU()
    elif activation_fun == "gelu":
        return nn.GELU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def layernorm(w_in):
    """Helper for building a layernorm layer."""
    return nn.LayerNorm(w_in, eps=1e-4)


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=1e-4, momentum=0.1)


def patchify2d(w_in, w_out, k, *, bias=True):
    """Helper for building a patchify layer as used by ViT models."""
    return nn.Conv2d(w_in, w_out, k, stride=k, padding=0, bias=bias)


# Copied from timm
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ViTHead(nn.Module):
    """Transformer classifier, an fc layer."""

    def __init__(self, w_in, num_classes):
        super().__init__()
        self.head_fc = linear(w_in, num_classes, bias=True)

    def forward(self, x):
        return self.head_fc(x)


class MLPBlock(nn.Module):
    """Transformer MLP block, fc, gelu, fc."""

    def __init__(self, w_in, mlp_d):
        super().__init__()
        self.linear_1 = linear(w_in, mlp_d, bias=True)
        self.af = activation("gelu")
        self.linear_2 = linear(mlp_d, w_in, bias=True)

    def forward(self, x):
        return self.linear_2(self.af(self.linear_1(x)))


class ViTEncoderBlock(nn.Module):
    """Transformer encoder block, following https://arxiv.org/abs/2010.11929."""

    def __init__(self, hidden_d, n_heads, mlp_d):
        super().__init__()
        self.ln_1 = layernorm(hidden_d)
        self.self_attention = MultiheadAttention(hidden_d, n_heads)
        # # NOTE: PyTorch original MHA module causes graph break under TorchDynamo,
        # # so use our own impl of MHA for now.
        # self.self_attention = Attention(hidden_d, num_heads=n_heads)
        self.ln_2 = layernorm(hidden_d)
        self.mlp_block = MLPBlock(hidden_d, mlp_d)

    def forward(self, x):
        x_p = self.ln_1(x)
        # x_p = self.self_attention(x_p)
        x_p, _ = self.self_attention(x_p, x_p, x_p)
        x = x + x_p
        x_p = self.mlp_block(self.ln_2(x))
        return x + x_p


class ViTEncoder(nn.Module):
    """Transformer encoder (sequence of ViTEncoderBlocks)."""

    def __init__(self, n_layers, hidden_d, n_heads, mlp_d):
        super(ViTEncoder, self).__init__()
        for i in range(n_layers):
            self.add_module(f"block_{i}", ViTEncoderBlock(hidden_d, n_heads, mlp_d))
        self.ln = layernorm(hidden_d)

    def forward(self, x):
        for block in self.children():
            x = block(x)
        return x


class ViTStemPatchify(nn.Module):
    """The patchify vision transformer stem as per https://arxiv.org/abs/2010.11929."""

    def __init__(self, w_in, w_out, k):
        super(ViTStemPatchify, self).__init__()
        self.patchify = patchify2d(w_in, w_out, k, bias=True)

    def forward(self, x):
        return self.patchify(x)


class ViT(nn.Module):
    """Vision transformer as per https://arxiv.org/abs/2010.11929."""

    def check_params(params):
        p = params
        err_str = "Input shape indivisible by patch size"
        assert p["image_size"] % p["patch_size"] == 0, err_str
        assert p["stem_type"] in ["patchify", "conv"], "Unexpected stem type"
        assert p["cls_type"] in ["token", "pooled"], "Unexpected classifier mode"
        if p["stem_type"] == "conv":
            err_str = "Conv stem layers mismatch"
            assert len(p["c_stem_dims"]) == len(p["c_stem_strides"]), err_str
            assert len(p["c_stem_strides"]) == len(p["c_stem_kernels"]), err_str
            err_str = "Stem strides unequal to patch size"
            assert p["patch_size"] == np.prod(p["c_stem_strides"]), err_str
            err_str = "Stem output dim unequal to hidden dim"
            assert p["c_stem_dims"][-1] == p["hidden_d"], err_str

    def __init__(self, params, has_breakpoint=False):
        super(ViT, self).__init__()
        p = params
        ViT.check_params(p)
        self.stem = ViTStemPatchify(3, p["hidden_d"], p["patch_size"])
        seq_len = (p["image_size"] // p["patch_size"]) ** 2
        class_token_len = 1  # NOTE: make class token length to be multiple of 8, to work better with Alpa
        self.class_token = nn.Parameter(torch.zeros(1, class_token_len, p["hidden_d"]))
        seq_len += class_token_len
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, p["hidden_d"]))
        self.encoder = ViTEncoder(
            p["n_layers"], p["hidden_d"], p["n_heads"], p["mlp_d"]
        )
        self.head = ViTHead(p["hidden_d"], p["num_classes"])

    def forward(self, x):
        # (n, c, h, w) -> (n, hidden_d, n_h, n_w)
        x = self.stem(x)
        # (n, hidden_d, n_h, n_w) -> (n, hidden_d, (n_h * n_w))
        x = x.reshape(x.size(0), x.size(1), -1)
        # (n, hidden_d, (n_h * n_w)) -> (n, (n_h * n_w), hidden_d)
        x = x.permute(0, 2, 1)
        if self.class_token is not None:
            # Expand the class token to the full batch
            class_token = self.class_token.expand(x.size(0), -1, -1)
            x = torch.cat([class_token, x], dim=1)
        x = x + self.pos_embedding
        x = self.encoder(x)
        # `token` or `pooled` features for classification
        x = x[:, 0, :] if self.class_token is not None else x.mean(dim=1)
        x = self.head(x)
        return x


def weight_init_func(pt_module, name_map, params, bufs):
    # for k, m in pt_module.named_modules():
    #     if isinstance(m, nn.Linear):
    #         params[name_map[f"{k}.weight"]] = nn.init.xavier_uniform(
    #             params[name_map[f"{k}.weight"]])
    #         params[name_map[f"{k}.bias"]] = nn.init.normal(
    #             params[name_map[f"{k}.bias"]], std=1e-6)
    return params, bufs


class TorchViTTest(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(123)
        alpa.set_seed(123)

    def test_vit_pipeshard(self):
        batch_size = 16
        num_channels = 3
        image_size = 224
        patch_size = 14
        num_classes = 8

        vit_params = {
            "image_size": image_size,
            "patch_size": patch_size,
            "stem_type": "patchify",
            "n_layers": 2,
            "n_heads": 8,
            "hidden_d": 8,
            "mlp_d": 32,
            "cls_type": "token",
            "num_classes": num_classes,
            "c_stem_kernels": [],
            "c_stem_strides": [],
            "c_stem_dims": [],
        }
        pt_module_gen = lambda: ViT(params=vit_params)

        dataloader = [
            (torch.randn(batch_size, num_channels, image_size, image_size), torch.randn(batch_size, num_classes)),
            (torch.randn(batch_size, num_channels, image_size, image_size), torch.randn(batch_size, num_classes)),
        ]
        loss_func = lambda *args, **kwargs: nn.functional.mse_loss(
            *args, **kwargs)
        optim_gen = torchoptim.adam(lr=1e-3)
        num_micro_batches = 2

        parallel_method = alpa.PipeshardParallel(
            stage_mode="auto", num_micro_batches=num_micro_batches)

        # _xla_client_mem_fraction_orig_value = alpa.global_config.xla_client_mem_fraction
        # alpa.global_config.xla_client_mem_fraction = 0.7

        train_torch_module(pt_module_gen, weight_init_func, dataloader,
                           loss_func, optim_gen, parallel_method)

        # alpa.global_config.xla_client_mem_fraction = _xla_client_mem_fraction_orig_value


def suite():
    suite = unittest.TestSuite()
    suite.addTest(TorchViTTest("test_vit_pipeshard"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
