"""
.. _Alpa Quickstart:

Alpa Quickstart
===============

Alpa is built on top of a tensor computation framework `Jax <https://jax.readthedocs.io/en/latest/index.html>`_ .
Alpa can automatically parallelize jax functions and runs them on a distributed cluster.
Alpa analyses the computational graph and generates a distributed execution plan 
tailored for the computational graph and target cluster.
The generated execution plan can combine state-of-the-art distributed training techniques
including data parallelism, operator parallelism, and pipeline parallelism.

Alpa provides a simple API ``alpa.parallelize`` and automatically generates the best execution
plan by solving optimization problems. Therefore, you can efficiently scale your jax computation
on a distributed cluster, without any expertise in distributed computing.

In this tutorial, we show the usage of Alpa with an MLP example.
"""

################################################################################
# Import Libraries
# ----------------
# We first import the required libraries.
# Flax and optax are libraries on top of jax for training neural networks.
# Although we use these libraries in this example, Alpa works on jax's and XLA's internal
# intermediate representations and does not depend on any specific high-level libraries.

from functools import partial

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
# parallelize it later. We train an MLP to learn a function y = Wx + b.

class MLPModel(nn.Module):
    hidden_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        for i in range(self.num_layers):
            if i % 2 == 0:
                x = nn.Dense(features=self.hidden_dim * 4)(x)
            else:
                x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.relu(x)
        return x

dim = 2048
batch_size = 2048
num_layers = 10

# Generate ground truth W and b
rngkey = jax.random.PRNGKey(0)
k1, k2 = random.split(rngkey)
W = random.normal(k1, (dim, dim), jnp.float32)
b = random.normal(k2, (dim,), jnp.float32)

# Generate the training data
ksample, knoise = random.split(k1)
x = random.normal(ksample, (batch_size, dim), jnp.float32)
y = (x @ W + b) + 0.1 * random.normal(knoise, (batch_size, dim))

# Initialize a train state, which includes the model paramter and optimizer state.
model = MLPModel(hidden_dim=dim, num_layers=num_layers)
params = model.init(rngkey, x)
tx = optax.adam(learning_rate=1e-3)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

# Define the training function and execute one step
def train_step(state, batch):
    def loss_func(params):
        out = state.apply_fn(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

batch = {"x": x, "y": y}
expected_state = train_step(state, batch)

################################################################################
# Auto-parallelization with ``alpa.parallelize``
# ----------------------------------------------
# Alpa provides a transformation ``alpa.parallelize`` to parallelize a jax function.
# ``alpa.parallelize`` is similar to ``jax.jit`` . ``jax.jit`` compiles a jax
# function for a single device, while ``alpa.parallelize`` compiles a jax function
# for a distributed device cluster.
# You may know that jax has some built-in transformations for parallelization,
# such as ``pmap``, ``pjit``, and ``xmap``. However, these transformations are not
# fully automatic, because they require users to manually specify the parallelization
# strategies such as parallelization axes and device mapping schemes. You also need to
# manually call communication primitives such as ``lax.pmean`` and ``lax.all_gather``,
# which is nontrivial if you want to do advanced model parallelization.
# Unlike these transformations, ``alpa.parallelize`` can do all things automatically for
# you. ``alpa.parallelize`` finds the best parallelization strategy for the given jax
# function and does the code tranformation. You only need to write the code as if you are
# writing for a single device.

# Define the training step. The body of this function is the same as the
# ``train_step`` above. The only difference is to decorate it with
# ``alpa.paralellize``.

@alpa.parallelize
def alpa_train_step(state, batch):
    def loss_func(params):
        out = model.apply(params, batch["x"])
        loss = jnp.mean((out - batch["y"])**2)
        return loss

    grads = jax.grad(loss_func)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return new_state

# Test correctness
actual_state = alpa_train_step(state, batch)
assert_allclose(expected_state.params, actual_state.params, atol=5e-3)

################################################################################
# After being decorated by ``alpa.parallelize``, the function can still take numpy
# arrays or jax arrays as inputs. The function will first distribute the input
# arrays into correct devices according to the parallelization strategy and then
# execute the function distributedly. The returned result arrays are also
# stored distributedly.

print("Input parameter type:", type(state.params["params"]["Dense_0"]["kernel"]))
print("Output parameter type:", type(actual_state.params["params"]["Dense_0"]["kernel"]))

# Get one copy
kernel_np = np.array(actual_state.params["params"]["Dense_0"]["kernel"])

################################################################################
# Execution Speed Comparison
# --------------------------
# By parallelizing a jax function, we can accelerate the computation and reduce
# the memory usage per GPU, so we can train large models faster.
# We benchmark the execution speed of ``jax.jit`` and ``alpa.parallelize``
# on a 8-GPU machine.

state = actual_state  # We need this assignment because the original `state` is "donated" and freed.
from alpa.util import benchmark_func

# Benchmark serial execution with jax.jit
jit_train_step = jax.jit(train_step, donate_argnums=(0,))

def sync_func():
    jax.local_devices()[0].synchronize_all_activity()

def serial_execution():
    global state
    state = jit_train_step(state, batch)

costs = benchmark_func(serial_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3

print(f"Serial execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

# Benchmark parallel execution
# We distribute arguments in advance for the benchmarking purpose
state, batch = alpa_train_step.preshard_dynamic_args(state, batch)

def parallel_execution():
    global state
    state = alpa_train_step(state, batch)

costs = benchmark_func(parallel_execution, sync_func, warmup=5, number=10, repeat=5) * 1e3

print(f"Parallel execution time. Mean: {np.mean(costs):.2f} ms, Std: {np.std(costs):.2f} ms")

################################################################################
# Memory Usage Comparison
# -----------------------
# We can also compare the memory usage per GPU.

GB = 1024 ** 3

executable = jit_train_step.lower(state, batch).compile().runtime_executable()
print(f"Serial execution per GPU memory usage: {executable.total_allocation_size() / GB:.2f} GB")

executable = alpa_train_step.get_executable(state, batch)
print(f"Parallel execution per GPU memory usage: {executable.get_total_allocation_size() / GB:.2f} GB")
