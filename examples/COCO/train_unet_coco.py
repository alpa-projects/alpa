import functools
from typing import Tuple

import flax.linen as nn
import jax.numpy as jnp
import time

import jax
from flax.training import train_state

import numpy as np 
import optax 


from typing import Any, Callable, Sequence, Tuple
import operator
# from coco_dataset import Coco2014
# from dataset import dataloader
import tensorflow as tf
from functools import partialmethod
import utils
import argparse
import tqdm
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

# TODO(hexu): Now we are using dummy inputs, later we could enable this function for testing on real data. 
# def get_data(batch_size):
#     """Load COCO train and test datasets into memory."""
#     dataset = Coco2014('/l/users/hexu.zhao/coco_dataset')
#     train_loader, valid_loader = dataloader(dataset, batch_size=batch_size)
#     return dataset, train_loader, valid_loader

##########################  Train ##############################

def train_step_jit(params, opt_state, batch):
    """Train for a single step."""

    def training_loss_fn(params, batch):
        unet_out = model.apply(
            params, 
            batch[0], 
            train=True)
        label_one_hot = jax.nn.one_hot(batch[1], config["labels_count"])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=unet_out, labels=label_one_hot))
        return loss
    
    grads = alpa.grad(training_loss_fn)(params, batch)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

@parallelize(batch_argnums=(2,))
def train_step_alpa(params, opt_state, batch):
    """Train for a single step."""

    def training_loss_fn(params, batch):
        unet_out = model.apply(
            params, 
            batch[0], 
            train=True)
        label_one_hot = jax.nn.one_hot(batch[1], config["labels_count"])
        loss = jnp.mean(optax.softmax_cross_entropy(logits=unet_out, labels=label_one_hot))
        return loss

    training_loss_fn = automatic_layer_construction(training_loss_fn, layer_num=4)
    grads = alpa.grad(training_loss_fn)(params, batch)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

# TODO(hexu): Now we are using dummy inputs, later we could enable this function for testing on real data. 
# def train_epoch(state, train_loader, epoch, rng):
#     """Train for a single epoch."""
#     # losses = []
#     # time_spend = []
#     for batch in train_loader:
#         # begin_time = time.time()
#         state = train_step(state, (jnp.array(batch[0]), jnp.array(batch[1])))
#         # end_time = time.time()
#         # losses.append(loss)
#         # time_spend.append(end_time-begin_time)
#         # if len(losses) % 100 == 0:
#         #     print('train epoch: %d, step %d, loss: %.4f time: %.2f sec' %
#         #           (epoch, len(losses), np.mean(losses), end_time-begin_time))
#     # print(f"average time per batch: {np.mean(time_spend)} sec")
#     # print(f"time spend per epoch: {np.sum(time_spend)} sec")
#     return state

def create_train_state(rngkey, model, input_args, learning_rate_fn):
    params = model.init_dummy(rngkey, **input_args)

    num_trainable_params = utils.compute_param_number(params)
    model_size = utils.compute_bytes(params)
    print(f"model size : {model_size/utils.MB} MB, num of trainable params : {num_trainable_params/1024/1024} M")

    global tx
    tx = optax.sgd(learning_rate=learning_rate_fn)
    opt_state = tx.init(params)
    return params, opt_state

########################## Main Scripts ##############################

memory_choices = [37] # unit GB
gpu_choices = [1, 2, 4, 8, 16, 32]
channel_size_choices = [(16, 32, 64, 128), (16, 64, 64, 128), (32, 64, 64, 128), (32, 64, 128, 256), (16, 32, 64, 128, 256, 512), (32, 32, 128, 128, 512, 512)]
block_cnt_choices = [(4, 4, 4, 4, 4), (6, 6, 6, 6, 6), (6, 6, 6, 6, 6), (8, 8, 8, 8, 8), (12, 12, 12, 12, 12, 12), (12, 12, 12, 12, 12, 12)]

config = {
    "padding": "SAME",
    "use_batch_norm": False,
    "channel_size": (16, 32, 64, 128), # (64, 128, 256, 512)
    "block_cnt": (4, 4, 4, 4, 4),
    "batch_size": 64,
    "learning_rate": 0.3,
    "tf_seed": 0,
    "image_h": 384,
    "image_w": 384,
    "labels_count": 12,
    "train_epochs": 1
}
model=None
tx=None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpu", choices=[2,4], type=int, required=True)
    args = parser.parse_args()
    mode = gpu_choices.index(args.num_gpu)
    config["channel_size"] = channel_size_choices[mode]
    config["block_cnt"] = block_cnt_choices[mode]

    ray.init(address="auto", namespace=get_ray_namespace_str(prefix="alpa-unittest"))
    tf.config.experimental.set_visible_devices([], 'GPU')
    jax.config.update('jax_platform_name', 'cpu')
    device_cluster = DeviceCluster()
    devices = device_cluster.get_virtual_physical_mesh()
    rngkey = jax.random.PRNGKey(0)
    set_parallelize_options(devices=devices, strategy="pipeshard_parallel", pipeline_stage_mode="auto_gpipe")

    print(config)
    tf.random.set_seed(config["tf_seed"])
    print(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    print(f'JAX local devices: {jax.local_devices()}')

    disable_tqdm_globally()

    # create dataset
    # TODO(hexu): Now we are using dummy inputs, later we could enable this function for testing on real data. 
    # dataset, train_loader, valid_loader = get_data(config['batch_size'])

    # initialize model
    model = UNet(num_classes=config["labels_count"],
                padding=config['padding'],
                use_batch_norm=config['use_batch_norm'],
                channel_size=config['channel_size'],
                block_cnt=config['block_cnt'])

    rng = jax.random.PRNGKey(0)
    dummy_input = {
        "x": jnp.zeros((config["batch_size"], config["image_h"], config["image_w"], 3), jnp.float32),
        "train": False
    }
    params, opt_state = create_train_state(rng, model, dummy_input, config["learning_rate"])

    batch = (dummy_input["x"], jnp.ones((config["batch_size"], config["image_h"], config["image_w"])))
    params_jit, opt_state_jit = train_step_jit(params, opt_state, batch)
    print("finish jit input")
    params_alpa, opt_state_alpa = train_step_alpa(params, opt_state, batch)
    print("finish alpa input")
    
    assert_allclose(params_jit, params_alpa, 1e-3, 1e-3)

    num_epochs = 10
    for _ in range(num_epochs):
        params_jit, opt_state_jit  = train_step_jit(params_jit, opt_state_jit, batch)
        params_alpa, opt_state_alpa = train_step_alpa(params_alpa, opt_state_alpa, batch)
    
    assert_allclose(params_jit, params_alpa, 1e-3, 1e-3)
    train_step_alpa.get_executable(params_alpa, opt_state_alpa, batch).shutdown()
