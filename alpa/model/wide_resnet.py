"""The definition of wide-resnet.

Modified from https://github.com/google/flax/blob/main/examples/imagenet/models.py.
see also: https://arxiv.org/pdf/1605.07146.pdf
"""
# Copyright 2021 The Flax Authors.
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
# limitations under the License.

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax.numpy as jnp

ModuleDef = Any


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    width_factor: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(
        self,
        x,
    ):
        assert self.width_factor == 1

        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides,
                                 name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    width_factor: int
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * self.width_factor, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides,
                                 name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int
    width_factor: int
    dtype: Any = jnp.float32
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(nn.Conv, use_bias=False, dtype=self.dtype)
        norm = partial(nn.BatchNorm,
                       use_running_average=not train,
                       momentum=0.9,
                       epsilon=1e-5,
                       dtype=self.dtype)

        x = conv(self.num_filters, (7, 7), (2, 2),
                 padding=[(3, 3), (3, 3)],
                 name='conv_init')(x)
        x = norm(name='bn_init')(x)
        x = nn.relu(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2**i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   width_factor=self.width_factor,
                                   act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


model_configs = {
    0: {
        "stage_sizes": [],
        "block_cls": ResNetBlock
    },
    18: {
        "stage_sizes": [2, 2, 2, 2],
        "block_cls": ResNetBlock
    },
    34: {
        "stage_sizes": [3, 4, 6, 3],
        "block_cls": ResNetBlock
    },
    50: {
        "stage_sizes": [3, 4, 6, 3],
        "block_cls": BottleneckResNetBlock
    },
    101: {
        "stage_sizes": [3, 4, 23, 3],
        "block_cls": BottleneckResNetBlock
    },
    152: {
        "stage_sizes": [3, 8, 36, 3],
        "block_cls": BottleneckResNetBlock
    },
    200: {
        "stage_sizes": [3, 24, 36, 3],
        "block_cls": BottleneckResNetBlock
    }
}


def get_wide_resnet(num_layers, width_factor, num_filters, num_classes, dtype):
    model_config = model_configs[num_layers]
    model_config["width_factor"] = width_factor
    model_config["num_filters"] = num_filters
    model_config["num_classes"] = num_classes
    model_config["dtype"] = dtype

    return ResNet(**model_config)
