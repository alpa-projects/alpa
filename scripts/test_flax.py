import functools
import time
from typing import Any, Callable, Sequence, Optional

import numpy as np
import jax
from jax import lax, random, numpy as jnp
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from flax import optim

N = 2 ** 8
H = N


class Model(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Dense(features=H, use_bias=False)(x)
    x = nn.Dense(features=H, use_bias=False)(x)
    return x


@flax.struct.dataclass
class TrainState:
    optimizer: optim.Optimizer
    model_state: Any

model = Model()

def create_train_state(batch):
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, batch['x'])
    optimizer = optim.Momentum().create(params)
    return TrainState(optimizer, None)


def train_step(state, batch):
    def loss_func(params):
        out = model.apply(params, batch['x'])
        return jnp.mean((out - batch['y']) ** 2)

    grad = jax.grad(loss_func)(state.optimizer.target)
    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=0.1)

    return TrainState(new_optimizer, None)


def get_train_step_jaxpr(create_func, train_step_func, batch):
    def whole_graph(batch):
        init_state = jax.jit(create_func)(batch)
        new_state = jax.jit(train_step_func)(init_state, batch)
        return new_state
    return jax.make_jaxpr(whole_graph)(batch).jaxpr.eqns[1]

batch = {"x": jnp.ones((N, H)), "y": jnp.ones((N, H))}

train_graph = get_train_step_jaxpr(create_train_state, train_step, batch)
print(train_graph)

