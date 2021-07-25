import jax
import jax.numpy as jnp
import numpy as np
import os
import ray
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax
from jax.experimental.maps import FrozenDict as FrozenDictJax

from parax import parallelize, set_parallelize_options, mark_pipeline, DeviceCluster

MB = 1024 ** 2
num_gpus = 2
# in order for ray to work we have to set this
# so the driver program and actor program can share GPUs...
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# jax.config.update('jax_platform_name', 'cpu')
# assert len(jax.local_devices()) >= num_gpus
# devices = tuple(jax.local_devices()[:num_gpus])

def is_sequence(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True

def assert_allclose(x, y):
    if isinstance(x, dict) or isinstance(x, FrozenDictJax) or isinstance(x, FrozenDictFlax):
        assert isinstance(y, dict) or isinstance(y, FrozenDictJax) or isinstance(x, FrozenDictFlax)
        assert set(x.keys()) == set(y.keys())
        for k in x.keys():
            assert_allclose(x[k], y[k])
    elif is_sequence(x) and not hasattr(x, '__array__'):
        assert is_sequence(y) and not hasattr(y, '__array__')
        assert len(x) == len(y)
        for x_elt, y_elt in zip(x, y):
            assert_allclose(x_elt, y_elt)
    elif hasattr(x, '__array__') or np.isscalar(x):
        assert hasattr(y, '__array__') or np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        assert np.allclose(x, y)
    elif x == y:
        return
    else:
        raise TypeError((type(x), type(y)))


class Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # FIXME (zhuohan): if don't require the gradient of x here, the
        #                  backward pass of the pipeline start will not
        #                  be generated.
        x, = mark_pipeline(x, name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
        x = nn.relu(x)
        x, = mark_pipeline(x, name='1', mark_type='end')
        x, = mark_pipeline(x, name='2', mark_type='start')
        x = nn.Dense(features=self.output_dim, use_bias=False)(x)
        return x

def train_step(optimizer, batch, apply_fn):
    def loss_func(params, x, y):
        out = apply_fn(params, x)
        loss = jnp.mean((out - y) ** 2)
        loss, = mark_pipeline(loss, name='2', mark_type='end')
        return loss

    grad_param, grad_x = jax.grad(loss_func, argnums = (0, 1))(optimizer.target, batch['x'], batch['y'])
    # FIXME (zhuohan): make the pipeline work with apply_gradient
    # new_optimizer = optimizer.apply_gradient(grad_param)
    return grad_param


ray.init(address="auto", ignore_reinit_error=True)

device_cluster = DeviceCluster()
mesh = device_cluster.get_virtual_mesh()
batch_size = 128
hidden_dim = 2048
input_dim = output_dim = hidden_dim

x = jnp.ones((batch_size, input_dim))
y = jnp.ones((batch_size, output_dim))

# Init model and optimizer
model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)

gradients = train_step(optimizer, {"x": x, "y": y}, model.apply)
strategy = "3d_parallel"
assert_allclose(x, y)


set_parallelize_options(devices=mesh, strategy=strategy)
# pipelined_train_step = parallelize(donate_argnums=(), devices=mesh, strategy=strategy)(train_step)
pipelined_train_step = parallelize(donate_argnums=())(train_step)
gradients_with_pipeline = pipelined_train_step(optimizer, {"x": x, "y": y}, model.apply)
assert_allclose(gradients, gradients_with_pipeline)
