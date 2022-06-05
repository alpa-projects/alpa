from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import optax

from alpa import parallelize, ShardParallel, CreateStateParallel

def test_parallel_plan():
    use_bias = False
    batch_size = 64
    input_dim = output_dim = hidden_dim = 128

    class Model(nn.Module):

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=hidden_dim, use_bias=use_bias)(x)
            x = nn.relu(x)
            x = nn.Dense(features=output_dim, use_bias=use_bias)(x)
            x = nn.relu(x)
            return x

    def train_step(state, batch):

        def loss_func(params):
            out = state.apply_fn(params, batch["x"])
            return jnp.mean((out - batch["y"])**2)

        grads = jax.grad(loss_func)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state

    def create_state():
        model = Model()
        rngkey = jax.random.PRNGKey(0)
        params = model.init(rngkey, jnp.ones((1, input_dim)))
        tx = optax.adam(learning_rate=1e-2)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    batch = {
        "x": jnp.ones((batch_size, input_dim)),
        "y": jnp.ones((batch_size, output_dim)),
    }

    train_step = parallelize(train_step, method=ShardParallel())
    create_state = parallelize(create_state, method=CreateStateParallel(train_step, batch))

    state = create_state()
    state = train_step(state, batch)


if __name__ == "__main__":
    test_parallel_plan()
