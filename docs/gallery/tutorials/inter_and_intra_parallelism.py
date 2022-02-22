"""
Distributed Training with Both Intra- and Inter-Operator Parallelism
====================================================================

Alpa can automatically parallelizes jax functions with both intra-operator
parallelism (e.g. data parallelism, tensor-model parallelism) and inter-operator
parallelism (e.g. pipeline parallelism). The :ref:`getting started guide
<Getting Started with Alpa>`. focuses on using Alpa for intra-operator
parallelism.

In this tutorial, we will show how to use Alpa to parallelize an MLP model with
both intra- and inter-operator parallelism. First we will show how to use Alpa
to manually assign stages for inter-operator parallelism. Then we will show how
to use Alpa to automate this process.
"""

################################################################################
# Import Libraries and Initialize Environment
# ----------------------------------------------
# We first import the required libraries.

import alpa
from alpa.testing import assert_allclose
from flax import linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import optax

################################################################################
# Besides of alpa and jax related libraries, we also import `ray <https://docs.
# ray.io/>`_and start (or connect to) a ray cluster. We use ray to manage the
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
# ------------------------
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
# ------------------------------------------
# To manually assign stages for inter-operator parallelism, we can use the
# ``alpa.mark_pipeline`` function to mark the start and end of each pipeline stage,
# and use the ``@alpa.manual_layer_construction`` decorator to indicate that we
# are manually assigning stages.

# Define the model
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


# Initialize a train state, which includes the model paramter and optimizer state.
manual_intra_model = ManualIntraMLPModel(hidden_dim=dim)
manual_intra_state = TrainState.create(apply_fn=manual_intra_model.apply, params=params, tx=tx)

# Define training step
@alpa.parallelize
def manual_intra_train_step(state, batch):
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

    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

updated_manual_state = manual_intra_train_step(manual_intra_state, batch)
