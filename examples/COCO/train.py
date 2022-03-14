import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import time

import jax
from flax.training import train_state
import numpy as np 
import optax 
from absl import logging
from clu import metric_writers
from clu import periodic_actions
import ml_collections
from typing import Any, Callable, Sequence, Tuple
import operator
from coco_dataset import Coco2014, dataloader
import tensorflow as tf
from functools import partial
import utils
import argparse
import tqdm
import random
from unet import UNet
from alpa.model.model_util import TrainState
from alpa import (parallelize, set_parallelize_options, mark_pipeline,
                  DeviceCluster, automatic_layer_construction)

from alpa.testing import assert_allclose
from alpa.util import get_ray_namespace_str
import ray
import alpa
from alpa.util import disable_tqdm_globally

# The code is to test U-net with alpa 
# Command : python train_unet_coco.py --num-gpu 2

##########################  dataset ############################

def get_data(batch_size, path):
    """Load COCO train and test datasets into memory."""
    dataset = Coco2014(path)
    train_loader, valid_loader = dataloader(dataset, batch_size=batch_size)
    return dataset, train_loader, valid_loader

##########################  Train ##############################


def get_train_step(model, tx, layer_num=8, labels_count=12):
    
    @parallelize(batch_argnums=(2,))
    def train_step_alpa(params, opt_state, batch):
        """Train for a single step."""

        def training_loss_fn(params, batch):
            unet_out = model.apply(
                params, 
                batch[0], 
                train=True)
            label_one_hot = jax.nn.one_hot(batch[1], labels_count)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=unet_out, labels=label_one_hot))
            return loss

        training_loss_fn = automatic_layer_construction(training_loss_fn, layer_num=layer_num)
        grads = alpa.grad(training_loss_fn)(params, batch)
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state
    
    return train_step_alpa

def train_epoch(params, opt_state, train_loader, epoch, train_step):
    """Train for a single epoch."""
    losses = []
    time_spend = []
    for batch in train_loader:
        begin_time = time.time()
        params, opt_state = train_step(params, opt_state, (jnp.array(batch[0]), jnp.array(batch[1])))
        end_time = time.time()
        loss = -1
        losses.append(loss)
        time_spend.append(end_time-begin_time)
        if len(losses) % 100 == 0:
            print('train epoch: %d, step %d, loss: %.4f time: %.2f sec' %
                  (epoch, len(losses), np.mean(losses), end_time-begin_time))
    print(f"average time per batch: {np.mean(time_spend)} sec")
    print(f"time spend per epoch: {np.sum(time_spend)} sec")
    return params, opt_state

def create_train_state(rngkey, model, image_h, image_w, learning_rate_fn):
    dummy_input = {
        "x": jnp.zeros((1, image_h, image_w, 3), jnp.float32),
        "train": False
    }
    params = model.init_dummy(rngkey, **dummy_input)

    num_trainable_params = utils.compute_param_number(params)
    model_size = utils.compute_bytes(params)
    print(f"model size : {model_size/utils.MB} MB, num of trainable params : {num_trainable_params/1024/1024} M")

    tx = optax.sgd(learning_rate=learning_rate_fn)
    opt_state = tx.init(params)
    return params, opt_state, tx

########################## Main Scripts ##############################

def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
    """Execute model training and evaluation loop.

    Args:
        config: Hyperparameter configuration for training and evaluation.
        workdir: Directory where the tensorboard summaries are written to.

    Returns:
        Final TrainState.
    """

    ray.init(address="auto", namespace=get_ray_namespace_str(prefix="alpa-unittest"))
    tf.config.experimental.set_visible_devices([], 'GPU')
    jax.config.update('jax_platform_name', 'cpu')
    if config.batch_size % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    device_cluster = DeviceCluster()
    devices = device_cluster.get_virtual_physical_mesh()
    # devices = alpa.PhysicalDeviceMesh()
    set_parallelize_options(devices=devices, strategy="pipeshard_parallel", pipeline_stage_mode="auto_gpipe")
    disable_tqdm_globally()

    rng = jax.random.PRNGKey(0)
    local_batch_size = config.batch_size // jax.process_count()

    # initialize model
    model = UNet(num_classes=config.labels_count,
                padding=config.padding,
                use_batch_norm=config.use_batch_norm,
                channel_size=config.channel_size,
                block_cnt=config.block_cnt)

    params, opt_state, tx = create_train_state(rng, model, config.image_h, config.image_w, config.learning_rate)

    # create dataset
    dataset, train_loader, _ = get_data(config.batch_size, config.data_path)

    train_step = get_train_step(model, tx, config.layer_num, config.labels_count)
    for epoch in range(config.train_epochs):
        params, opt_state = train_epoch(params, opt_state, train_loader, epoch, train_step)

    batch = (jnp.ones((config.batch_size, config.image_h, config.image_w, config.labels_count), 
                      type=np.float32), jnp.ones((config.batch_size, config.image_h, config.image_w), type=np.int32))
    train_step.get_executable(params, opt_state, batch).shutdown()
