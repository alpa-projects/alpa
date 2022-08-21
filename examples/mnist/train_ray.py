# Copyright 2022 The Flax Authors.
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

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import time


from absl import logging
import alpa
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
import tensorflow_datasets as tfds


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x


@alpa.parallelize
def train_step(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = state.apply_fn({'params': params}, images)
    one_hot = jax.nn.one_hot(labels, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  state = state.apply_gradients(grads=grads)
  return state, loss, accuracy


@alpa.parallelize(donate_argnums=())
def eval_step(state, images, labels):
  logits = state.apply_fn({'params': state.params}, images)
  one_hot = jax.nn.one_hot(labels, 10)
  loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  return loss, accuracy


def train_epoch(state, train_data_loader, steps_per_epoch):
  """Train for a single epoch."""
  epoch_loss = []
  epoch_accuracy = []

  for i in range(steps_per_epoch):
    batch_images, batch_labels = next(train_data_loader)
    state, loss, accuracy = train_step(state, batch_images, batch_labels)
    epoch_loss.append(loss)
    epoch_accuracy.append(accuracy)
  alpa.prefetch((epoch_loss, epoch_accuracy))
  train_loss = np.mean(epoch_loss)
  train_accuracy = np.mean(epoch_accuracy)
  return state, train_loss, train_accuracy


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = np.float32(train_ds['image']) / 255.
  test_ds['image'] = np.float32(test_ds['image']) / 255.
  train_ds['label'] = np.int32(train_ds['label'])
  test_ds['label'] = np.int32(test_ds['label'])
  return train_ds, test_ds


def create_train_state(rng, config):
  """Creates initial `TrainState`."""
  cnn = CNN()
  params = cnn.init(rng, jnp.ones([1, 28, 28, 1]))['params']
  tx = optax.sgd(config.learning_rate, config.momentum)
  return train_state.TrainState.create(
      apply_fn=cnn.apply, params=params, tx=tx)


def get_train_data_laoder(train_ds, state, batch_size):
  images_np = train_ds['image']
  labels_np = train_ds['label']
  steps_per_epoch = len(images_np) // batch_size

  def input_iter_func(start, end, batch_size):
    while True:
      for i in range(steps_per_epoch):
        idx = start + i * batch_size
        yield (images_np[idx:idx + batch_size],
               labels_np[idx:idx + batch_size])

  batch_images = jax.core.ShapedArray(
      (batch_size, 28, 28, 1), jnp.float32)
  batch_labels = jax.core.ShapedArray(
      (batch_size,), jnp.int32)
  executable = train_step.get_executable(state, batch_images, batch_labels)

  data_loader = alpa.MeshDriverDataLoader(
      batch_size, len(images_np),
      input_iter_func, executable.get_input_placement_specs()[1:3],
      prefetch_size=4, repeat=True)
  return iter(data_loader), steps_per_epoch


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  alpa.init(cluster="ray")
  train_ds, test_ds = get_datasets()

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng = jax.random.PRNGKey(0)
  state = create_train_state(rng, config)

  train_data_loader, steps_per_epoch = get_train_data_laoder(
      train_ds, state, config.batch_size)

  for epoch in range(1, config.num_epochs + 1):
    tic = time.time()
    state, train_loss, train_accuracy = train_epoch(state, train_data_loader,
                                                    steps_per_epoch)
    epoch_time = time.time() - tic
    test_loss, test_accuracy = eval_step(state, test_ds['image'], test_ds['label'])
    test_accuracy = np.array(test_accuracy)
    logging.info(
        'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f, epoch_time: %.3f'
        % (epoch, train_loss, train_accuracy * 100, test_loss,
           test_accuracy * 100, epoch_time))

    summary_writer.scalar('train_loss', train_loss, epoch)
    summary_writer.scalar('train_accuracy', train_accuracy, epoch)
    summary_writer.scalar('test_loss', test_loss, epoch)
    summary_writer.scalar('test_accuracy', test_accuracy, epoch)

  summary_writer.flush()
  return state
