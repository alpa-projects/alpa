import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.training import train_state
import jax
from flax import linen as nn, optim
from typing import Any, Callable, Sequence, Tuple


########################## Unet model ############################

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))

class TrainState(train_state.TrainState):
    dynamic_scale: optim.DynamicScale

class IdentityLayer(nn.Module):
    """Identity layer, convenient for giving a name to an array."""

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x

def central_crop(inputs, target_shape):
    """Returns a central crop in axis (1, 2).

    Args:
        inputs: nd-array; Inputs in shape of `[bs, height, width, channels]'.
        target_shape: tuple(int); Target shape after crop.

    Returns:
        Cropped image.
    """
    h, w = target_shape[1:3]
    assert h <= inputs.shape[1], f'{h} > {inputs.shape[1]}'
    assert w <= inputs.shape[2], f'{w} > {inputs.shape[2]}'
    h0 = (inputs.shape[1] - h) // 2
    w0 = (inputs.shape[2] - w) // 2
    return inputs[:, h0:(h0 + h), w0:(w0 + w)]


class DeConv3x3(nn.Module):
    """Deconvolution layer for upscaling.

    Attributes:
        features: Num convolutional features.
        padding: Type of padding: 'SAME' or 'VALID'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        """Applies deconvolution with 3x3 kernel."""
        if self.padding == 'SAME':
            padding = ((1, 2), (1, 2))
        elif self.padding == 'VALID':
            padding = ((0, 0), (0, 0))
        else:
            raise ValueError(f'Unkonwn padding: {self.padding}')
        x = nn.Conv(features=self.features,
                    kernel_size=(3, 3),
                    input_dilation=(2, 2),
                    padding=padding)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class ConvRelu2(nn.Module):
    """Two unpadded convolutions & relus.

  Attributes:
    features: Num convolutional features.
    convs: Num convolutional layers.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    convs: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        for i in range(self.convs):
            x = Conv3x3(features=self.features, name='conv'+str(i),
                        padding=self.padding)(x)
            if self.use_batch_norm:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
        return x


class DownsampleBlock(nn.Module):
    """Two unpadded convolutions & downsample 2x.

  Attributes:
    features: Num convolutional features
    convs: Num convolutional layers
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    convs: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        residual = x = ConvRelu2(features=self.features,
                                 convs=self.convs,
                                 padding=self.padding,
                                 use_batch_norm=self.use_batch_norm)(
                                     x, train=train)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, residual


class BottleneckBlock(nn.Module):
    """Two unpadded convolutions, dropout & deconvolution.

  Attributes:
    features: Num convolutional features.
    convs: Num convolutional layers.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    convs: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = ConvRelu2(self.features,
                      convs=self.convs,
                      padding=self.padding,
                      use_batch_norm=self.use_batch_norm)(x, train=train)
        x = DeConv3x3(features=self.features // 2,
                      name='deconv',
                      padding=self.padding,
                      use_batch_norm=self.use_batch_norm)(x, train=train)
        return x


class UpsampleBlock(nn.Module):
    """Two unpadded convolutions and upsample.

    Attributes:
        features: Num convolutional features.
        convs: Num convolutional layers.
        padding: Type of padding: 'SAME' or 'VALID'.
        use_batch_norm: Whether to use batchnorm at the end or not.
    """

    features: int
    convs: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual, *,
                 train: bool) -> jnp.ndarray:
        if residual is not None:
            x = jnp.concatenate([x, central_crop(residual, x.shape)], axis=-1)
        x = ConvRelu2(self.features,
                      convs=self.convs,
                      padding=self.padding,
                      use_batch_norm=self.use_batch_norm)(x, train=train)
        x = DeConv3x3(features=self.features // 2,
                      name='deconv',
                      padding=self.padding,
                      use_batch_norm=self.use_batch_norm)(x, train=train)
        return x


class OutputBlock(nn.Module):
    """Two unpadded convolutions followed by linear FC.

  Attributes:
    features: Num convolutional features.
    num_classes: Number of classes.
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    convs: int
    num_classes: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = ConvRelu2(self.features,
                      convs=self.convs,
                      padding=self.padding,
                      use_batch_norm=self.use_batch_norm)(x, train=train)
        x = nn.Conv(features=self.num_classes,
                    kernel_size=(1, 1),
                    name='conv1x1')(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        return x


class UNet(nn.Module):
    """U-Net from http://arxiv.org/abs/1505.04597.

  Based on:
  https://github.com/NVIDIA/DeepLearningExamples/blob/master/TensorFlow2/Segmentation/UNet_Medical/model/unet.py
  Note that the default configuration `config.padding="VALID"` does only work
  with images that have a certain minimum size (e.g. 128x128 is too small).

  Attributes:
    num_classes: Number of classes.
    channel_size: Sequence of feature sizes used in UNet blocks.
    block_cnt: The number of conv2 at each height. In original version UNet, they are all 2.
    padding: Type of padding.
    use_batch_norm: Whether to use batchnorm or not.
  """

    num_classes: int
    channel_size: Tuple[int, ...] = (64, 128, 256, 512)
    block_cnt: Tuple[int, ...] = (2, 2, 2, 2)
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 *,
                 train: bool,
                 debug: bool = False) -> jnp.ndarray:
        """Applies the UNet model."""
        del debug
        skip_connections = []
        for i, (features,convs) in enumerate(zip(self.channel_size, self.block_cnt[:-1])):
            x, residual = DownsampleBlock(features=features,
                                          convs=convs,
                                          padding=self.padding,
                                          use_batch_norm=self.use_batch_norm,
                                          name=f'0_down_{i}')(x, train=train)
            skip_connections.append(residual)
        x = BottleneckBlock(features=2 * self.channel_size[-1],
                            convs=self.block_cnt[-1],
                            padding=self.padding,
                            use_batch_norm=self.use_batch_norm,
                            name='1_bottleneck')(x, train=train)

        *upscaling_features, final_features = self.channel_size[::-1]
        *upscaling_convs, final_convs = self.block_cnt[:-1][::-1]
        
        for i, (features, convs) in enumerate(zip(upscaling_features, upscaling_convs)):
            x = UpsampleBlock(features=features,
                              convs=convs,
                              padding=self.padding,
                              use_batch_norm=self.use_batch_norm,
                              name=f'2_up_{i}')(
                                  x,
                                  residual=skip_connections.pop(),
                                  train=train)

        x = IdentityLayer(name='pre_logits')(x)
        x = OutputBlock(features=final_features,
                        convs=final_convs,
                        num_classes=self.num_classes,
                        padding=self.padding,
                        use_batch_norm=self.use_batch_norm,
                        name='output_projection')(x, train=train)
        return x

    def init_dummy(self, *args, **kwargs):
        return self.init(
            *args, **kwargs
        )  # batchnorm will cause ConcretizationTypeError(Any method to fix this?), thus here I use concrete value forward.

        avals = jax.eval_shape(self.init, *args, **kwargs)
        return jax.tree_util.tree_map(
            lambda x: jnp.full(x.shape, 1e-8, x.dtype), avals)
