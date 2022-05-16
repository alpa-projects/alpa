"""
Distributed Training with Both Shard and Pipeline Parallelism
=============================================================

Alpa can automatically parallelizes jax functions with both shard
parallelism (a.k.a. intra-operator parallelism) and pipeline parallelism
(a.k.a. inter-operator parallelism). Shard parallelism includes
data parallelism, operator parallelism, and their combinations.
The :ref:`quick start <Alpa Quickstart>` focuses on using Alpa for shard parallelism.

In this tutorial, we show how to use Alpa with both shard and pipeline parallelism.
First, we show how to use Alpa to manually assign stages for pipeline parallelism.
Then we show how to use Alpa to automate this process.
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
import ray

################################################################################
# Connect to a Ray Cluster
# -------------------------------------------
# Alpa uses a distributed framework `ray <https://docs.ray.io/>`_ to manage
# the cluster and disributed workers. We initialize ray and alpa.

ray.init()
alpa.init(cluster="ray")

# Alternatively, you can use the following command to connect to an existing
# ray cluster.
# ray.init(address="auto")

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
W = random.normal(k1, (dim, dim), jnp.float32)
b = random.normal(k2, (dim,), jnp.float32)

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim), jnp.float32)
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim), jnp.float32)

# Initialize a train state, which includes the model paramter and optimizer
# state.
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
# Pipeline Parallelism with Manual Assignment
# -------------------------------------------
# To manually assign stages for pipeline parallelism, we can use the
# ``alpa.mark_pipeline`` function to mark the start and end of each pipeline
# stage, and use the ``@alpa.manual_layer_construction`` decorator to indicate
# that we are manually assigning stages. Note that each the pipeline stage is
# also automatically parallelized by the shard parallel pass.


# Define the manually parallelized model with pipeline markers.
class ManualPipelineMLPModel(nn.Module):
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

# Initialize the train state with the same parameters as the single-device
# model.
manual_pipeline_model = ManualPipelineMLPModel(hidden_dim=dim)
manual_pipeline_state = TrainState.create(apply_fn=manual_pipeline_model.apply,
                                          params=copy.deepcopy(params), tx=tx)

# Define the training step with manually parallelized pipeline stages.
# We use the "alpa.PipeshardParallel" option to let alpa use both
# pipeline parallelism and shard parallelism.
@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16))
def manual_pipeline_train_step(state, batch):
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

manual_pipeline_actual_state = manual_pipeline_train_step(manual_pipeline_state,
                                                          batch)
assert_allclose(expected_state.params, manual_pipeline_actual_state.params,
                atol=5e-3)

alpa.shutdown()

################################################################################
# Pipeline Parallelism with Automatic Assignment
# ----------------------------------------------
# Alpa also supports automatically partitioning the model into multiple
# pipeline stages and assign each pipeline stage a device mesh such that
# the total execution latency is minimized. Specifically, the automatic
# partitioning algorithm consists of the following steps:
#
# 1. **Layer Construction:** In this step, the operators in the model are
#    clustered into "layers" based on a graph clustering algorithm. The
#    user needs to specify the total number of layers (i.e. clusters) as
#    a hyperparameter.
# 2. **Stage Construction and Mesh Slicing:** In this step, we partition
#    the device cluster (device mesh) to multiple submeshes and assign
#    layers to submeshes to form pipeline stages to minimize the total
#    pipeline execution latency.

alpa.init(cluster="ray")

# Define training step with automatic pipeline-operator parallelism. Note that
# we reuse the same model and state as the single device case. The only
# modification required is the two decorators. The stage construction and
# mesh slicing are performed within the `parallelize` decorator.

@alpa.parallelize(method=alpa.PipeshardParallel(num_micro_batches=16, stage_mode="auto"))
def auto_pipeline_train_step(state, batch):
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

# In the first call, alpa triggers the compilation.
# The compilation first profiles several costs and solves an optimization
# problem to get the optimal pipeline assignments.
auto_pipeline_actual_state = auto_pipeline_train_step(state, batch)
assert_allclose(expected_state.params, auto_pipeline_actual_state.params,
                atol=5e-3)

alpa.shutdown()
