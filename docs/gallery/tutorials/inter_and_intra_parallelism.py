"""
Distributed Training with Both Intra- and Inter-Operator Parallelism
====================================================================

Alpa can automatically parallelizes jax functions with both intra-operator
parallelism (e.g. data parallelism, tensor-model parallelism) and inter-operator
parallelism (e.g. pipeline parallelism). The :ref:`getting started guide
<Getting Started with Alpa>`. focuses on using Alpa for intra-operator
parallelism.

In this tutorial, we show how to use Alpa to parallelize an MLP model with
both intra- and inter-operator parallelism. First, we show how to use Alpa
to manually assign stages for inter-operator parallelism. Then we show how
to use Alpa to automate this process.
"""

################################################################################
# Import Libraries and Initialize Environment
# -------------------------------------------
# We first import the required libraries.

import alpa
from alpa.testing import assert_allclose
import copy
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import optax

################################################################################
# Besides alpa and jax related libraries, we also import `ray <https://docs.
# ray.io/>`_ and start (or connect to) a ray cluster. We use ray to manage the
# devices in the distributed cluster in alpa.

import ray

ray.init()

# Alternatively, you can use the following command to connect to an existing
# ray cluster.
# ray.init(address="auto")

################################################################################
# In alpa, the actual computation of a computational graph is executed on ray
# actors. Therefore, we force the driver process to use the CPU to avoid it
# from occupying the GPU memory.
jax.config.update('jax_platform_name', 'cpu')

################################################################################
# Train an MLP on a Single Device
# -------------------------------
# In this tutorial, we use a toy dataset to train an MLP model.
# Specifically, we use the model to fit the function: :math:`y = Wx + b`.
# Note that now this model is being executed on CPU because we force the driver
# process to use the CPU.

class MLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x

dim = 2048
batch_size = 2048

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer state.
model = MLPModel(hidden_dim=dim)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define training step
def train_step(state, batch):
    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {"x": x, "y": y}
expected_state = train_step(state, batch)

################################################################################
# Manual Inter-Operator Parallelism with Alpa
# -------------------------------------------
# To manually assign stages for inter-operator parallelism, we can use the
# ``alpa.mark_pipeline`` function to mark the start and end of each pipeline stage,
# and use the ``@alpa.manual_layer_construction`` decorator to indicate that we
# are manually assigning stages. Note that all the pipeline stages are also
# automatically parallelized by the intra-operator parallel pass.

# Set the number of microbatches for pipeline parallelism.
num_micro_batches = 16

# Initialize the alpa device cluster.
device_cluster = alpa.DeviceCluster()
devices = device_cluster.get_virtual_physical_mesh()

# Set the parallel strategy to "pipeshard_parallel" to enable both inter- and intra-
# operator parallelism.
alpa.set_parallelize_options(
    devices=devices, strategy="pipeshard_parallel",
    num_micro_batches=num_micro_batches)

# Define the manually parallelized model with pipeline markers.
class ManualIntraMLPModel(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        # Mark the end of the 0th pipeline stage and the start of the 1st
        # pipeline stage. the start marker of the 0th stage and the end
        # marker of the 1st stage are marked in the train_step below.
        alpa.mark_pipeline(name='0', mark_type='end')
        alpa.mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim * 4)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.relu(x)
        return x

# Initialize the train state with the same parameters as the single-device model.
manual_inter_model = ManualIntraMLPModel(hidden_dim=dim)
manual_inter_state = TrainState.create(apply_fn=manual_inter_model.apply,
                                       params=copy.deepcopy(params), tx=tx)

# Define the training step with manually parallelized pipeline stages.
@alpa.parallelize
def manual_inter_train_step(state, batch):
    # Indicate that we are manually assigning pipeline stages.
    @alpa.manual_layer_construction
    def loss_func(params):
        # Mark the start of the 0th pipeline stage.
        alpa.mark_pipeline(name='0', mark_type='start')
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        # Mark the end of the 1st pipeline stage.
        alpa.mark_pipeline(name='1', mark_type='end')
        return loss

    # We use `alpa.grad` here to seperate the apply gradient stage with the
    # forward/backward stages in the pipeline. This is necessary to ensure that
    # the gradient accumulation is correct.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

manual_inter_actual_state = manual_inter_train_step(manual_inter_state, batch)
assert_allclose(expected_state.params, manual_inter_actual_state.params, atol=5e-3)

# Terminate the alpa device cluster.
manual_inter_train_step.get_executable(manual_inter_state, batch).shutdown()

################################################################################
# Automatic Inter-Operator Parallelism with Alpa
# ----------------------------------------------
# Alpa also supports automatically partitioning the model into multiple
# pipeline stages and assign each pipeline stage a device mesh such that
# the total execution latency is minimized. Specifically, the automatic
# partitioning algorithm consists of the following steps:
#
# 1. **Layer Construction:** In this step, the operators in the model are
#    clustered into ``layers'' based on a graph clustering algorithm. The
#    user needs to specify the total number of layers (i.e. clusters) as
#    a hyperparameter.
# 2. **Stage Construction and Mesh Slicing:** In this step, we partition
#    the device cluster (device mesh) to multiple submeshes and assign
#    layers to submeshes to form pipeline stages to minimize the total
#    pipeline execution latency.


# Create a new cluster class for automatic inter-operator parallelism.
device_cluster = alpa.DeviceCluster()
devices = device_cluster.get_virtual_physical_mesh()
# Set pipeline stage mode to "auto_gpipe" to enable automatic inter-operator
# parallelism with automatic stage slicing and mesh assignment.
alpa.set_parallelize_options(
    devices=devices, strategy="pipeshard_parallel", pipeline_stage_mode="auto_gpipe",
    num_micro_batches=num_micro_batches)

# Define training step with automatic inter-operator parallelism. Note that
# we reuse the same model and state as the single device case. The only
# modification required is the two decorators. The stage construction and
# mesh slicing are performed within the `parallelize` decorator.
@alpa.parallelize
def auto_inter_train_step(state, batch):
    # Indicate that we use automatic layer construction. The `layer_num` here
    # is a hyperparameter to control how many layers we get from the
    # layer construction algorithm.
    @alpa.automatic_layer_construction(layer_num=2)
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    # Again, we use `alpa.grad` here to seperate the apply gradient stage with
    # the forward/backward stages in the pipeline.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

auto_inter_actual_state = auto_inter_train_step(state, batch)
assert_allclose(expected_state.params, auto_inter_actual_state.params, atol=5e-3)

auto_inter_train_step.get_executable(state, batch).shutdown()
