from functools import partial 

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental.maps import Mesh, mesh, xmap
from jax.lax import pdot, pmean, psum
from jax.nn import relu


def test_dist_matmul():
    func = xmap(
        jnp.vdot,
        in_axes=({0: 'left'}, {1: 'right'}),
        out_axes=['left', 'right', ...],
        axis_resources={'left': 'x', 'right': 'y'})

    devices = np.array(jax.devices())[:4].reshape((2, 2))
    with mesh(devices, ('x', 'y')):  # declare a 2D mesh with axes 'x' and 'y'
        x = jnp.arange(20).reshape((4, 5))
        out = func(x, x.T)

        print(out.sharding_spec)


def test_collective_pdot():
    def f(x, y):
        return pdot(x, y, 'k')

    x = jnp.ones((3, 4))
    y = jnp.ones((4, 5))
    z = jax.pmap(f, axis_name='k', in_axes=(1, 0), out_axes=None)(x, y)

    print(z.sharding_spec)


def test_mlp():
    def loss_func(x, y, w1, w2):
        x = relu(pdot(x, w1, 'model'))
        x = relu(pdot(x, w2, 'hidden'))
        loss = (x - y) ** 2
        loss = psum(loss, 'model')
        loss = pmean(loss, 'batch')
        return loss

    serial_step = xmap(
        loss_func,
        in_axes=({0: 'batch', 1: 'model'},
                 {0: 'batch', 1: 'model'},
                 {0: 'model', 1: 'hidden'},
                 {0: 'model', 1: 'hidden'},),
        out_axes={})

    parallel_step = xmap(
        loss_func,
        in_axes=({0: 'batch', 1: 'model'},
                 {0: 'batch', 1: 'model'},
                 {0: 'model', 1: 'hidden'},
                 {0: 'model', 1: 'hidden'},),
        out_axes={},
        axis_resources={'batch': 'data_parallel',
                        'hidden': 'model_parallel'})

    x  = jnp.ones((8, 256))
    y  = jnp.ones((8, 256))
    w1 = jnp.ones((256, 1024))
    w2 = jnp.ones((256, 1024))

    serial_out = serial_step(x, y, w1, w2)

    data_parallel = 2
    model_parallel = 2
    devices = np.array(jax.devices())[:4].reshape((data_parallel, model_parallel))
    with mesh(devices, ('data_parallel', 'model_parallel')):
        parallel_out = parallel_step(x, y, w1, w2)

        print(parallel_out.sharding_spec)


def test_grad():
    def loss(x, y):
        loss = (x - y) ** 2
        loss = pmean(loss, 'batch')
        return loss


    loss_parallel = xmap(
        loss,
        in_axes=({0: 'batch'},
                 {0: 'batch'},),
        out_axes={},
        axis_resources={'batch': 'i'})

    x = jnp.ones((16,))
    y = jnp.ones((16,))

    devices = np.array(jax.devices()[:4])
    with mesh(devices, ('i',)):
        # out = loss_parallel(x, y)
        # print(out.sharding_spec)

        grad = jax.grad(loss_parallel)(x, y)


if __name__ == "__main__":
    test_dist_matmul()
    #test_collective_pdot()
    #test_mlp()
    #test_grad()

