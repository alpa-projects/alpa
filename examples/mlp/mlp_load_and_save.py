import unittest
import os

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from jax import random
import optax
import ray
import pickle
from tempfile import TemporaryFile

import alpa
from alpa import (parallelize, set_parallelize_options, mark_pipeline,
                  DeviceCluster, automatic_layer_construction)
from alpa.model.model_util import TrainState
from alpa.testing import assert_allclose
from alpa.util import get_ray_namespace_str, tree_to_nparray

### For this quickstart tutorial, we'll be running linear regression on a 4-layer MLP. 

### Initialize ray and devices
# We initialize ray and a device cluster to run the model on. In addition, 
# we also initialize a RNG key that will be used later for initializing parameters.
ray.init(address="auto", namespace=get_ray_namespace_str(prefix="alpa-unittest"))
jax.config.update('jax_platform_name', 'cpu')
device_cluster = DeviceCluster()
devices = device_cluster.get_virtual_physical_mesh()
rngkey = jax.random.PRNGKey(0)

### Set global environment variables
# Here, we're setting the devices used to be the device cluster we defined earlier.
# set_parallelize_options(devices=devices, strategy="pipeshard_parallel")
set_parallelize_options(devices=devices, strategy="pipeshard_parallel", pipeline_stage_mode="auto_stage")

### Building our model
# The model will be a two layer MLP model with a ReLU non-linearity. The input an output
# dimensions stay the same. 
class OurModel(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        x = nn.relu(x)
        return x
        

### Generate inputs and outputs. 
# Let's run a regression task, where we try to have the model learn y=Wx+b. 
batch_size = 64
nsamples = 20
dim = 1024

# Generate random ground truth W and b
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Create the predict function from a set of parameters
def make_predict(W,b):
  def predict(x):
    return jnp.dot(W,x)+b
  return predict

# Generate samples with additional noise
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = jax.vmap(make_predict(W, b))(x) + 0.1*random.normal(knoise,(batch_size, dim))


### Initialize training parameters. 
# We instantiate our model and initialize the parameters randomly. 
# We use Flax for training, where the parameters of a model are stored seperately from the model.
model = OurModel(hidden_dim=dim, output_dim=dim)
params = model.init(rngkey, x)

# Next, we instantiate our SGD optimizer, and initialize the optimizer state. This state
# will be updated later on in the training loop.
tx = optax.sgd(learning_rate=1e-2)
opt_state = tx.init(params)

### Define the training loop. 
# We define a mean squared error loss function, and manually set it as its individual pipeline step.
# Layer boundaries will be set up at the input and outputs of the loss function.

@parallelize(batch_argnums=(2,))
def parallel_train_step(opt_state, params, batch):
    @automatic_layer_construction(layer_num=4)
    def loss_func(params, x, y):
        out = model.apply(params, x)
        loss = jnp.mean((out - y)**2)
        return loss

    grads = alpa.grad(loss_func)(params, batch["x"], batch["y"])
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

def train_step(opt_state, params, batch):
    def loss_func(params, x, y):
        out = model.apply(params, x)
        loss = jnp.mean((out - y)**2)
        return loss

    grads = alpa.grad(loss_func)(params, batch["x"], batch["y"])
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params

# Now, we run one step of training. For simplicity, we only run one batch of data during training. 
# To check that the parallelize function returns the same result as the non-parallelized version, 
# we run one step of training normally and one step of training with the train_step parallelized. 
batch = {"x": x, "y": y}
state_orig, params_orig = train_step(opt_state, params, batch)
state_with_pipeline, params_with_pipeline = parallel_train_step(opt_state, params, batch)

# We check that the parameters are the same. 
assert_allclose(params_orig, params_with_pipeline, 1e-3, 1e-3)

# Continue training for 10 epochs
num_epochs = 10
for _ in range(num_epochs):
    state_orig, params_orig  = train_step(state_orig, params_orig, batch)
    state_with_pipeline, params_with_pipeline = parallel_train_step(state_with_pipeline, params_with_pipeline, batch)

# We check that the parameters are the same. 
assert_allclose(params_orig, params_with_pipeline, 1e-3, 1e-3)

# Save initial model, original model, and the parallel model to a temporary file
outfile = TemporaryFile()
state_dict = flax.serialization.to_state_dict(opt_state)
pickle.dump(tree_to_nparray(state_dict), outfile)
outfile_param = TemporaryFile()
param_dict = flax.serialization.to_state_dict(params)
pickle.dump(param_dict, outfile_param)

outfile_orig = TemporaryFile()
state_dict_orig = flax.serialization.to_state_dict(state_orig)
pickle.dump(tree_to_nparray(state_dict_orig), outfile_orig)
outfile_param_orig = TemporaryFile()
param_dict_orig = flax.serialization.to_state_dict(params_orig)
pickle.dump(param_dict_orig, outfile_param_orig)

outfile_with_pipeline = TemporaryFile()
state_dict_with_pipeline = flax.serialization.to_state_dict(state_with_pipeline)
pickle.dump(tree_to_nparray(state_dict_with_pipeline), outfile_with_pipeline)
outfile_param_with_pipeline = TemporaryFile()
param_dict_with_pipeline = flax.serialization.to_state_dict(params_with_pipeline)
pickle.dump(tree_to_nparray(param_dict_with_pipeline), outfile_param_with_pipeline)

# Load models from the temporary file
outfile.seek(0)
loaded_state_dict = pickle.load(outfile)
loaded_state = flax.serialization.from_state_dict(
    opt_state, loaded_state_dict)
outfile.close()

outfile_param.seek(0)
loaded_param_dict = pickle.load(outfile_param)
loaded_param = flax.serialization.from_state_dict(
    params, loaded_param_dict)
outfile_param.close()

outfile_orig.seek(0)
loaded_state_dict_orig = pickle.load(outfile_orig)
loaded_state_orig = flax.serialization.from_state_dict(
    state_orig, loaded_state_dict_orig)
outfile_orig.close()

outfile_param_orig.seek(0)
loaded_param_dict_orig = pickle.load(outfile_param_orig)
loaded_param_orig = flax.serialization.from_state_dict(
    params_orig, loaded_param_dict_orig)
outfile_param_orig.close()

outfile_with_pipeline.seek(0)
loaded_state_dict_with_pipeline = pickle.load(outfile_with_pipeline)
loaded_state_with_pipeline = flax.serialization.from_state_dict(
    state_with_pipeline, loaded_state_dict_with_pipeline)
outfile_with_pipeline.close()

outfile_param_with_pipeline.seek(0)
loaded_param_dict_with_pipeline = pickle.load(outfile_param_with_pipeline)
loaded_param_with_pipeline = flax.serialization.from_state_dict(
    params_with_pipeline, loaded_param_dict_with_pipeline)
outfile_param_with_pipeline.close()

# Evaluate all three loaded models
def loss_func(params, x, y):
    out = model.apply(params, x)
    loss = jnp.mean((out - y)**2)
    return loss

loss = loss_func(loaded_param, batch["x"], batch["y"])
loss_orig = loss_func(loaded_param_orig, batch["x"], batch["y"])
loss_with_pipeline = loss_func(loaded_param_with_pipeline, batch["x"], batch["y"])

print(f"Initial Loss: {loss}")
print(f"Single-device Loss: {loss_orig}")
print(f"Parallelized Loss: {loss_with_pipeline}")

assert_allclose(loss_orig, loss_with_pipeline, 1e-3, 1e-3)
assert loss > loss_with_pipeline, "Error: Trained loss is more than initial loss."

# Shutting down the the pipelined executable. 
parallel_train_step.get_executable(opt_state, params, batch).shutdown()
