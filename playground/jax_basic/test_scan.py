from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim

batch_size = 32
hidden_size = 128

class Layer(nn.Module):
    @nn.compact
    def __call__(self, x):
        return

class Model(nn.Module):
    def __call__(self, x):
        cell = nn.scan(
            nn.Dense,
            variable_broadcast="params",
            in_axes=1,
            out_axes=1,
            split_rngs={"params": False},
        )

@partial(jax.jit, static_argnums=(2,))
def train_step(optimizer, batch, apply_fn):
    def loss_func(params):
        out = apply_fn(params, batch["x"])
        return jnp.mean((out - batch["y"]) ** 2)

    grad = jax.grad(loss_func)(optimizer.target)
    new_optimizer = optimizer.apply_gradient(grad)
    return new_optimizer

x = jnp.ones((batch_size, hidden_size))
y = jnp.ones((batch_size, hidden_size))

# Init model and optimizer
model = Model()
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)

# JIT compile
optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

