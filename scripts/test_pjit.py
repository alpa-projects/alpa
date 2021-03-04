from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.nn import relu
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint


def test_basic1d():
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
        return x + y

    x = np.ones((8, 8))

    mesh_devices = np.array(jax.devices()[:2])
    with mesh(mesh_devices, ('x',)):
        actual = f(x, x + 1)


def test_matmul():
    @partial(pjit,
             in_axis_resources=(P(None, 'x'), P('x', None)),
             out_axis_resources=None)
    def f(x, y):
        return x @ y

    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:2])
    with mesh(mesh_devices, ('x',)):
        out = f(x, y)

    np.testing.assert_allclose(out, x @ y, rtol=1e-5)


def test_mlp_forward():
    def loss_func(batch, weights):
        x, y = batch
        w1, w2 = weights

        x = x @ w1
        x = relu(x)
        #x = with_sharding_constraint(x, P('data_parallel', 'model_parallel'))
        x = x @ w2
        x = relu(x)
        loss = jnp.mean((x - y) ** 2)
        return loss

    loss_func_parallel = pjit(
        loss_func,
        in_axis_resources=((P('data_parallel', None), P('data_parallel', None)),
                           (P(None, 'model_parallel'), P('model_parallel', None))),
        out_axis_resources=None,
    )

    N = 8
    D = 128

    np.random.seed(1)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    loss_serial = loss_func((x, y), (w1, w2))

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        loss_parallel = loss_func_parallel((x, y), (w1, w2))

    np.testing.assert_allclose(loss_serial, loss_parallel, rtol=1e-5)


def test_mlp_grad():
    def loss_func(batch, weights):
        x, y = batch
        w1, w2 = weights

        x = x @ w1
        x = relu(x)
        x = with_sharding_constraint(x, P('data_parallel', 'model_parallel'))
        x = x @ w2
        x = relu(x)
        loss = jnp.mean((x - y) ** 2)
        return loss

    def step_serial(batch, weights):
        gradients = jax.grad(loss_func, argnums=1)(batch, weights)
        return (w - g * lr for w, g in zip(weights, gradients))

    step_parallel = pjit(
        step_serial,
        in_axis_resources=((P('data_parallel', None), P('data_parallel', None)),
                           (P(None, 'model_parallel'), P('model_parallel', None))),
        out_axis_resources=((P(None, 'model_parallel'), P('model_parallel', None))),
    )

    lr = 0.1
    N = 8
    D = 128

    np.random.seed(1)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = step_serial((x, y), (w1, w2))

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        w1_parallel, w2_parallel = step_parallel((x, y), (w1, w2))


if __name__ == "__main__":
    #test_basic1d()
    #test_matmul()

    test_mlp_forward()
    #test_mlp_grad()

