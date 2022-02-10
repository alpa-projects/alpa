import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp

import jax
from flax.training import train_state

import numpy as np 
import optax 
import tensorflow_datasets as tfds

from typing import Any, Callable, Sequence, Tuple
import operator


# The code is to test U-net on minst. The task is identity mapping that inputs an image and outputs the same one.
# Training command : python train_unet_mnist.py



config = {
    "padding": "SAME",
    "use_batch_norm": True,
    "block_size": (8, 16),
    "batch_size": 1,
    "learning_rate": 0.01
}

##########################  model ############################

Conv3x3 = functools.partial(nn.Conv, kernel_size=(3, 3))

class TrainState(train_state.TrainState):
    batch_stats: Any


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
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = Conv3x3(features=self.features, name='conv1',
                    padding=self.padding)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = Conv3x3(features=self.features, name='conv2',
                    padding=self.padding)(x)
        if self.use_batch_norm:
            x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        return x


class DownsampleBlock(nn.Module):
    """Two unpadded convolutions & downsample 2x.

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
        residual = x = ConvRelu2(features=self.features,
                                 padding=self.padding,
                                 use_batch_norm=self.use_batch_norm)(
                                     x, train=train)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, residual


class BottleneckBlock(nn.Module):
    """Two unpadded convolutions, dropout & deconvolution.

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
        x = ConvRelu2(self.features,
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
    padding: Type of padding: 'SAME' or 'VALID'.
    use_batch_norm: Whether to use batchnorm at the end or not.
  """

    features: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, residual, *,
                 train: bool) -> jnp.ndarray:
        if residual is not None:
            x = jnp.concatenate([x, central_crop(residual, x.shape)], axis=-1)
        x = ConvRelu2(self.features,
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
    num_classes: int
    padding: str = 'SAME'
    use_batch_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, train: bool) -> jnp.ndarray:
        x = ConvRelu2(self.features,
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
    block_size: Sequence of feature sizes used in UNet blocks.
    padding: Type of padding.
    use_batch_norm: Whether to use batchnorm or not.
  """

    num_classes: int
    block_size: Tuple[int, ...] = (64, 128, 256, 512)
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
        for i, features in enumerate(self.block_size):
            x, residual = DownsampleBlock(features=features,
                                          padding=self.padding,
                                          use_batch_norm=self.use_batch_norm,
                                          name=f'0_down_{i}')(x, train=train)
            skip_connections.append(residual)
        x = BottleneckBlock(features=2 * self.block_size[-1],
                            padding=self.padding,
                            use_batch_norm=self.use_batch_norm,
                            name='1_bottleneck')(x, train=train)

        *upscaling_features, final_features = self.block_size[::-1]
        for i, features in enumerate(upscaling_features):
            x = UpsampleBlock(features=features,
                              padding=self.padding,
                              use_batch_norm=self.use_batch_norm,
                              name=f'2_up_{i}')(
                                  x,
                                  residual=skip_connections.pop(),
                                  train=train)

        x = IdentityLayer(name='pre_logits')(x)
        x = OutputBlock(features=final_features,
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


##########################  dataset ############################


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(
        ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))

    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


##########################  Train ##############################


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def training_loss_fn(params):
        unet_out, new_model_state = model.apply(
            {
                "params": params,
                "batch_stats": state.batch_stats
            },
            batch['image'],
            mutable=['batch_stats'],
            train=True)
        loss = ((unet_out - batch['image'])**2).mean()
        return loss, (new_model_state, unet_out)

    grad_fn = jax.value_and_grad(training_loss_fn, has_aux=True)
    (loss, (new_model_state, unet_out)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads,
                                  batch_stats=new_model_state["batch_stats"])
    return state, loss, unet_out


def train_epoch(state, train_ds, batch_size, epoch, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))
    losses = []
    for perm in perms:
        # print(f"start step {i}")
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, loss, unet_out = train_step(state, batch)
        losses.append(loss)
        if len(losses) % 500 == 0:
            print('train epoch: %d, step %d, loss: %.4f' %
                  (epoch, len(losses), np.mean(losses)))

    return state


def log_param_shapes(params, print_params_nested_dict: bool = False) -> int:
    """Prints out shape of parameters and total number of trainable parameters.

      Args:
        params: PyTree of model parameters.
        print_params_nested_dict: If True, it prints parameters in shape of a nested
          dict.

      Returns:
        int; Total number of trainable parameters.
    """

    total_params = jax.tree_util.tree_reduce(
        operator.add, jax.tree_util.tree_map(lambda x: x.size, params))
    print('Total params: %d', total_params)
    return total_params


def create_train_state(rngkey, model, input_args, learning_rate_fn):
    # print(input_args)
    params = model.init_dummy(rngkey, **input_args)
    params, batch_stats = params["params"], params["batch_stats"]

    print(jax.tree_map(lambda x: x.shape, params))
    print(jax.tree_map(lambda x: x.shape, batch_stats))

    num_trainable_params = log_param_shapes(params)  # compute number of param
    # dynamic_scale = optim.DynamicScale()
    dynamic_scale = None

    tx = optax.sgd(learning_rate=learning_rate_fn)
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats)
    return state

########################## Main Scripts ##############################

# create model
train_ds, test_ds = get_datasets()

# initialize model
model = UNet(num_classes=2,
             padding=config['padding'],
             use_batch_norm=config['use_batch_norm'],
             block_size=config['block_size'])

rng = jax.random.PRNGKey(0)
dummy_input = {
    "x": jnp.zeros((config['batch_size'], ) + train_ds['image'].shape[1:], jnp.float32),
    "train": False
}
trainstate = create_train_state(rng, model, dummy_input, config["learning_rate"])

num_epochs = 5
for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    trainstate = train_epoch(trainstate, train_ds, config['batch_size'], epoch, input_rng)
