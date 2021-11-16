from functools import partial

import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from jax.nn import relu
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint
from jax._src.random import _random_bits, threefry_2x32
import flax
from flax import linen as nn

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


def test_failed_matmul_case_1():
    # Case 1: SR = RR x SR
    @partial(pjit,
             in_axis_resources=(P(None, None), P('y', None)),
             out_axis_resources=P('x', None))
    def f(x, y):
        return x @ y

    x = np.random.randn(4, 128).astype(np.float32)
    y = np.random.randn(128, 4).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:4]).reshape((2, 2))
    with mesh(mesh_devices, ('x', 'y')):
        out = f(x, y)


def test_failed_matmul_case_2():
    # Case 2: SR = SR x SR
    @partial(pjit,
             in_axis_resources=(P('x', None), P('y', None)),
             out_axis_resources=P('x', None))
    def f(x, y):
        return x @ y

    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:4]).reshape((2, 2))
    with mesh(mesh_devices, ('x', 'y')):
        out = f(x, y)

    np.testing.assert_allclose(out, x @ y, rtol=1e-5)


def test_reduce_scatter():
    @partial(pjit,
             in_axis_resources=(P(None, 'x'), P('x', None)),
             out_axis_resources=P('x', None))
    def f(x, y):
        return x @ y

    x = np.random.randn(8, 4).astype(np.float32)
    y = np.random.randn(4, 8).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:2])
    with mesh(mesh_devices, ('x',)):
        out = f(x, y)

    np.testing.assert_allclose(np.array(out), x @ y, rtol=1e-5)


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


def test_random_bits():
    @partial(pjit,
             in_axis_resources=(P('x'), None),
             out_axis_resources=P('x'))
    def func(inputs, key):
      random_uniform = lax.rng_uniform(0.0, 1.0, inputs.shape)
      ret = inputs * random_uniform
      return ret

    inputs = jnp.ones((4096,))
    rngkey = jax.random.PRNGKey(0)

    mesh_devices = np.array(jax.devices()[:4])
    with mesh(mesh_devices, ('x',)):
        actual = func(inputs, rngkey)
        print(actual)
        actual = func(inputs, rngkey)
        print(actual)


# Monkey patch random generator to use stateful random generator.
# This can simplify the computational graph
def fast_uniform(key, shape, dtype, minval=0.0, maxval=1.0):
    shape = jax.core.as_named_shape(shape)
    return lax.rng_uniform(minval, maxval, shape.positional)

def remove_fold_in(key, data):
    return key

jax._src.random.uniform = fast_uniform
jax.random.uniform = fast_uniform
jax._src.random.fold_in = remove_fold_in
jax.random.fold_in = remove_fold_in


def test_dropout():
    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dropout(0.1, deterministic=False)(x)
            return x

    model = Model()

    @partial(pjit,
             in_axis_resources=(P('x', 'y', None), None),
             out_axis_resources=P('x', 'y', None))
    def func(inputs, key):
      ret = model.apply({}, inputs, rngs={"dropout": key})
      return ret

    inputs = jnp.ones((512, 512, 16))
    rngkey = jax.random.PRNGKey(0)

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('x', 'y')):
        actual = func(inputs, rngkey)
        #print(actual)


def test_embedding():
    vocab_size = 8192
    hidden_size = 768
    batch_size = 4
    seq_len = 128

    @partial(pjit,
             in_axis_resources=(P(None, 'y'), P('x', None)),
             out_axis_resources=P('x', None, 'y'))
    def func(embedding, inputs):
      ret = jnp.take(embedding, inputs, axis=0)
      return ret

    embedding = jnp.ones((vocab_size, hidden_size), dtype=np.float32)
    inputs = jnp.ones((batch_size, seq_len), dtype=np.int32)

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('x', 'y')):
        actual = func(embedding, inputs)


def test_all_to_all():
    @partial(pjit,
             in_axis_resources=P('x', 'y', None),
             out_axis_resources=P('x', None, 'y'))
    def f(x):
        return x

    x = np.random.randn(2, 2, 4).astype(np.float32)

    mesh_devices = np.array(jax.devices()[:4]).reshape(2, 2)
    with mesh(mesh_devices, ('x', 'y')):
        out = f(x)


if __name__ == "__main__":
    #test_basic1d()
    #test_matmul()
    #test_failed_matmul_case_1()
    #test_failed_matmul_case_2()
    #test_reduce_scatter()
    #test_matmul_speed()
    #test_dict_arg()

    #test_mlp_forward()
    #test_mlp_grad()

    #test_random_bits()
    #test_dropout()

    #test_embedding()

    test_all_to_all()

