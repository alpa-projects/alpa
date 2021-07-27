from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax import optim

batch_size = 32
hidden_size = 128

class Model(nn.Module):
    def setup(self):
        self.weight = self.param("weight",
                                 jax.nn.initializers.zeros, (hidden_size, hidden_size))

    def __call__(self, x):
        def cond_func(args):
            counter = args[0]
            return counter < 5

        def body_func(args):
            counter, x = args 
            return [counter + 1, x @ self.weight]

        return jax.lax.while_loop(cond_func, body_func, [0, x])[1]

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

