from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.nn import relu
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint

from util import benchmark_func


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
             in_axis_resources=(P('x', None), P('x', None)),
             out_axis_resources=P('x', None))
    def f(x, y):
        return x @ y

    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:2])
    with mesh(mesh_devices, ('x',)):
        out = f(x, y)

    np.testing.assert_allclose(out, x @ y, rtol=1e-5)


def split(a, axis):
    in_axis_resources = [None] * len(a.shape)
    in_axis_resources[axis] = 'x'

    split_func = pjit(lambda x: x,
                      in_axis_resources=P(*in_axis_resources),
                      out_axis_resources=P(*in_axis_resources))

    with mesh(np.array(jax.devices()), ('x',)):
        a = split_func(a)
    return a


def test_matmul_speed():
    N = M = 1024
    K = 1 << 19
    n_devices = len(jax.devices())

    x_jnp = jnp.empty((N, K), dtype=np.float32).block_until_ready()
    y_jnp = jnp.empty((K, M), dtype=np.float32).block_until_ready()

    @jax.jit
    def matmul(x, y):
        return x @ y

    def serial_func():
        z = matmul(x_jnp, y_jnp)
        z.block_until_ready()

    costs = benchmark_func(serial_func) * 1000
    print("Mean Cost: %.3f ms (std: %.3f ms)" % (np.mean(costs), np.std(costs)))

    x_split = split(x_jnp, 1).block_until_ready()
    y_split = split(y_jnp, 0).block_until_ready()

    parallel_matmul = pjit(matmul,
                           in_axis_resources=(P(None, 'x'), P('x', None)),
                           out_axis_resources=None)

    def parallel_func():
        z = parallel_matmul(x_split, y_split)
        z.block_until_ready()

    with mesh(np.array(jax.devices()), ('x',)):
        costs = benchmark_func(parallel_func) * 1000
    print("Mean Cost: %.3f ms (std: %.3f ms)" % (np.mean(costs), np.std(costs)))


def test_dict_arg():
    @partial(pjit,
             in_axis_resources=None,
             out_axis_resources=None)
    def f(inputs):
        x = inputs['x']
        y = inputs['y']
        return x @ y

    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:2])
    with mesh(mesh_devices, ('x',)):
        out = f({"x": x, "y": y})

    np.testing.assert_allclose(out, x @ y, rtol=1e-5)


def test_mlp_forward():
    def loss_func(batch, weights):
        x, y = batch
        w1, w2 = weights

        x = x @ w1
        x = relu(x)
        x = with_sharding_constraint(x, P('data_parallel', 'model_parallel'))
        x = x @ w2
        loss = x
        #x = relu(x)
        #loss = jnp.mean((x - y) ** 2)
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

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        loss_parallel = loss_func_parallel((x, y), (w1, w2))

    #loss_serial = loss_func((x, y), (w1, w2))
    #np.testing.assert_allclose(loss_serial, loss_parallel, rtol=1e-5)


def test_mlp_grad():
    def loss_func(batch, weights):
        x, y = batch
        w1, w2 = weights

        x = x @ w1
        x = with_sharding_constraint(x, P('data_parallel', 'model_parallel'))
        x = x @ w2
        loss = jnp.mean((x - y) ** 2)
        return loss

    def step_serial(batch, weights):
        gradients = jax.grad(loss_func, argnums=1)(batch, weights)
        return tuple(w - g for w, g in zip(weights, gradients))

    step_parallel = pjit(
        step_serial,
        in_axis_resources=((P('data_parallel', None), P('data_parallel', None)),
                           (P(None, 'model_parallel'), P('model_parallel', None))),
        out_axis_resources=((P(None, 'model_parallel'), P('model_parallel', None))),
    )

    step_serail = jax.jit(step_serial)

    lr = 1
    N = 256
    D = 8192

    np.random.seed(1)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        w1_parallel, w2_parallel = step_parallel((x, y), (w1, w2))

    #w1_serial, w2_serial = step_serial((x, y), (w1, w2))
    #np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-5)
    #np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-5)


if __name__ == "__main__":
    #test_basic1d()
    #test_matmul()
    #test_matmul_speed()
    #test_dict_arg()

    #test_mlp_forward()
    test_mlp_grad()

