"""
Getting Started with Alpa
=========================

Alpa is a library that automatically parallelizes jax functions and runs them
on a distributed cluster. Alpa analyses the jax computational graph and generates
a distributed execution plan tailored for the computational graph and target cluster.
Alpa is specifically designed for training large-scale neural networks.
The generated execution plan can combine state-of-the-art distributed training techniques
including data parallelism, operator parallelism, and pipeline parallelism.

Alpa provides a simple API ``@parallelize`` and automatically generates the best execution
plan by solving optimization problems. Therefore, you can efficiently scale your jax
computation to a distributed cluster, without any expertise in distributed computing. 

In this tutorial, we show the usage of Alpa with an MLP example.
"""

################################################################################
# Import Libraries
# --------------------
# We first import the required libraries.
# Flax and optax are libraries on top of jax for training neural networks.
# Although we use these libraries in this example, Alpa works at jax level and
# does not depend on these specific libraries.

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
# Train an MLP on a Single Device
# -------------------------------
# To begin with, we implement the model and training loop on a single device. We will
# parallelize it later. We train an MLP to learn the function y = Wx + b.

class MLPModel(nn.Module):
    hidden_dim: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        return x

dim = 4096
batch_size = 4096
num_layers = 8

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim))
b = random.normal(k2, (dim,))

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim))
y = (x @ W + b) + 0.1 * random.normal(knoise,(batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer state.
model = MLPModel(hidden_dim=dim, output_dim=dim, num_layers=num_layers)
params = model.init(rngkey, x)
tx = optax.sgd(learning_rate=1e-2)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define training step
def train_step(state, batch):
    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    # Note that we have to replace `jax.grad` with `alpa.grad` if we want
    # to use gradient accumulation and pipeline parallelism.
    grads = alpa.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {"x": x, "y": y}
expected_state = train_step(state, batch)

################################################################################
# Auto-parallelization with ``@parallelize``
# ------------------------------------------
# Alpa provides a transformation ``@alpa.parallelize`` to parallelize a jax function.
# ``@alpa.parallelize`` is similar to ``@jax.jit`` . ``@jax.jit`` compiles a jax
# function for a single device, while ``@alpa.parallelize`` compiles a jax function
# for a distributed device cluster.
# You may know that jax has some built-in transformations for parallelization,
# such as ``pmap``, ``pjit``, and ``xmap``. However, these transformations are not
# fully automatic, because they require users to manually specify the parallelization
# strategies such as parallelization axes and device mapping schemes. You also need to
# manually call communication primitives such as ``lax.pmean`` and ``lax.all_gather``,
# which is nontrivial if you want to do advanced model parallelization.
# Unlike these transformations, ``@alpa.parallelize` can do all things automatically for
# you. You only need to write the code as if you are writing for a single device.

# Transform the function and run it
parallel_train_step = alpa.parallelize(train_step)

actual_state = parallel_train_step(state, batch)

# Test correctness
assert_allclose(expected_state.params, actual_state.params)

# The types of parameters in actual_state become `ShardedDeviceArray`,
# which means the parameters are now stored distributedly on multiple devices.
print(type(actual_state.params["params"]["Dense_0"]["kernel"]))

################################################################################
# Speed Comparision 
# -----------------
# By parallelizing a jax function, we can accelerate the computation and reduce
# the memory usage per GPU, so we can train large models faster.
# We benchmark the execution speed of ``@jax.jit`` and ``@alpa.parallelize``
# on a 8-GPU machine.

from alpa.util import benchmark_func

jit_train_step = jax.jit(train_step, donate_argnums=(0,))

def sync_func():
    jax.local_devices()[0].synchronize_all_activity()

# Benchmark serial execution
def serial_execution():
    global state
    state = jit_train_step(state, batch)

costs = benchmark_func(serial_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3

print(f"Serial execution. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

# Benchmark parallel execution
def parallel_execution():
    global state
    state = parallel_train_step(state, batch)

costs = benchmark_func(parallel_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3

print(f"Parallel execution. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")
