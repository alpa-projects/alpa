from flax import linen as nn
from flax.training.train_state import TrainState
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import optax

from alpa import parallelize, ShardParallel


def create_distributed_train_state(create_state_fn, train_step, train_step_args):
    jaxpr, state_avals = make_jaxpr(create_state_fn, return_shape=True)()
    executable = train_step.get_executable(state_avals, *train_step_args)
    state_placement_info = executable.get_placement_info()[0]

    jaxprs, load_infos = slice_jaxpr(jaxpr, state_load_info)

    executables = compile_executables(jaxprs, load_infos)

    for executable in executables:
        arrays = executable()

    return arrays


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
        params = model.init(rngkey, x)
        tx = optax.adam(learning_rate=1e-2)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    batch = {
        "x": jnp.ones((batch_size, input_dim)),
        "y": jnp.ones((batch_size, output_dim)),
    }

    train_step = parallelize(train_step, method=ShardParallel())
    create_state = parallelize(create_state, method=CreateStateParallel(train_step, batch))

    state = create_state()
    state = train_step(state, {"x": x, "y": y})


if __name__ == "__main__":
    test_parallel_plan()
