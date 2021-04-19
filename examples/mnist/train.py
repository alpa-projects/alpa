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

"""MNIST example.

Library file which executes the training and evaluation loop for MNIST.
The data is loaded using tensorflow_datasets.
"""

# See issue #620.
# pytype: disable=wrong-keyword-args

import time
from functools import partial

from absl import logging
from flax import linen as nn
from flax import optim
from flax.metrics import tensorboard
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow_datasets as tfds

from parax import parallelize, annotate_gradient


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
    x = nn.log_softmax(x)
    return x


def get_initial_params(key):
  init_val = jnp.ones((1, 28, 28, 1), jnp.float32)
  initial_params = CNN().init(key, init_val)['params']
  return initial_params


def create_optimizer(params, learning_rate, beta):
  optimizer_def = optim.Momentum(learning_rate=learning_rate, beta=beta)
  optimizer = optimizer_def.create(params)
  return optimizer


def onehot(labels, num_classes=10):
  x = (labels[..., None] == jnp.arange(num_classes)[None])
  return x.astype(jnp.float32)


def cross_entropy_loss(logits, labels):
  return -jnp.mean(jnp.sum(onehot(labels) * logits, axis=-1))


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


#@jax.jit
@parallelize
def train_step(optimizer, batch):
  """Train for a single step."""
  def loss_fn(params):
    logits = CNN().apply({'params': params}, batch['image'])
    loss = cross_entropy_loss(logits, batch['label'])
    return loss, logits
  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, logits), grad = grad_fn(optimizer.target)
  grad = annotate_gradient(grad)
  optimizer = optimizer.apply_gradient(grad)
  metrics = compute_metrics(logits, batch['label'])
  return optimizer, metrics


@jax.jit
def eval_step(optimizer, batch):
  logits = CNN().apply({'params': optimizer.target}, batch['image'])
  return compute_metrics(logits, batch['label'])


def train_epoch(optimizer, train_ds, batch_size, epoch, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['image'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['image']))
  perms = perms[:steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))
  batch_metrics = []

  batch_times = []
  for perm in perms:
    batch = {k: jnp.array(v[perm, ...]).block_until_ready()
             for k, v in train_ds.items()}

    tic = time.time()
    optimizer, metrics = train_step(optimizer, batch)
    next(iter(metrics.values())).block_until_ready()
    toc = time.time()

    batch_metrics.append(metrics)
    batch_times.append(toc - tic)

  # compute mean of metrics across each batch in epoch.
  batch_metrics_np = jax.device_get(batch_metrics)
  epoch_metrics_np = {
      k: np.mean([metrics[k] for metrics in batch_metrics_np])
      for k in batch_metrics_np[0]}

  logging.info('train epoch: %d, loss: %.4f, accuracy: %.2f, time: %.2f',
	       epoch, epoch_metrics_np['loss'],
	       epoch_metrics_np['accuracy'] * 100, np.median(batch_times))

  return optimizer, epoch_metrics_np


def eval_model(optimizer, test_ds):
  metrics = eval_step(optimizer, test_ds)
  metrics = jax.device_get(metrics)
  summary = jax.tree_map(lambda x: x.item(), metrics)
  return summary['loss'], summary['accuracy']


def get_datasets():
  """Load MNIST train and test datasets into memory."""
  ds_builder = tfds.builder('mnist')
  ds_builder.download_and_prepare()
  train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
  test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
  train_ds['image'] = train_ds['image'] / 255.
  test_ds['image'] = test_ds['image'] / 255.

  #train_ds = {
  #  "image": np.ones((2*16384, 28, 28, 1), dtype=np.float32),
  #  "label": np.ones(2*16384, dtype=np.int32),
  #}
  #test_ds = {
  #  "image": np.ones((1024, 28, 28, 1), dtype=np.float32),
  #  "label": np.ones(1024, dtype=np.int32),
  #}

  return train_ds, test_ds


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str):
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    The trained optimizer.
  """
  train_ds, test_ds = get_datasets()
  rng = jax.random.PRNGKey(0)

  summary_writer = tensorboard.SummaryWriter(workdir)
  summary_writer.hparams(dict(config))

  rng, init_rng = jax.random.split(rng)
  params = get_initial_params(init_rng)
  optimizer = create_optimizer(
      params, config.learning_rate, config.momentum)

  for epoch in range(1, config.num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    optimizer, train_metrics = train_epoch(
        optimizer, train_ds, config.batch_size, epoch, input_rng)
    loss, accuracy = eval_model(optimizer, test_ds)

    logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                 epoch, loss, accuracy * 100)

    summary_writer.scalar('train_loss', train_metrics['loss'], epoch)
    summary_writer.scalar('train_accuracy', train_metrics['accuracy'], epoch)
    summary_writer.scalar('eval_loss', loss, epoch)
    summary_writer.scalar('eval_accuracy', accuracy, epoch)

  summary_writer.flush()
  return optimizer

