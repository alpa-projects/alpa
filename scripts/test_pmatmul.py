from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

def split(a, axis, factor):
    assert a.shape[axis] % factor == 0
    new_shape = a.shape[:axis] + (factor, a.shape[axis] // factor) + a.shape[axis+1:]
    a = a.reshape(new_shape)
    a = jax.pmap(lambda x: x, in_axes=axis, out_axes=axis)(a)
    return a

def replica(a, factor):
    a = jax.pmap(lambda x, y: x, in_axes=(None, 0), out_axes=None)(a, jnp.ones(factor))
    return a

def unsplit(a, axis):
    new_shape = a.shape[:axis] + (a.shape[axis] * a.shape[axis+1],) + a.shape[axis+2:]
    return a.reshape(new_shape)


def test_matmul_k_partition():
    def matmul_k_partition(lhs, rhs):
        @partial(jax.pmap,
                 axis_name='k',
                 in_axes=(1, 0),
                 out_axes=None)
        def matmul(lhs, rhs):
            res = lhs @ rhs
            return jax.lax.psum(res, axis_name='k')

        return matmul(lhs, rhs)

    a = jnp.ones((1024, 1024))
    b = jnp.ones((1024, 1024))

    a = split(a, 1)
    b = split(b, 0)
    c = matmul_k_partition(a, b)

    print(c.shape, c.sharding_spec)


def test_mlp_forward():
    @partial(jax.pmap, in_axes=(None, 1), out_axes=1)
    def matmul_r_s1_s1(x, w):
        return x @ w

    @partial(jax.pmap, in_axes=(1, 0), out_axes=None, axis_name='k')
    def matmul_s1_s0_r(x, w):
        res = x @ w
        return jax.lax.psum(res, axis_name='k')

    N = 1024
    D = 1024

    x = jnp.ones((N, D))
    w1 = jnp.ones((D, D))
    w2 = jnp.ones((D, D))

    x = replica(x)
    w1 = split(w1, axis=1)
    w2 = split(w2, axis=0)

    x = matmul_r_s1_s1(x, w1)
    x = matmul_s1_s0_r(x, w2)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def f_operator(x, axis_name):
    return x

def f_operator_fwd(x, axis_name):
    return f_operator(x), ()

def f_operator_bwd(axis_name, res, g):
    return jax.lax.psum(x, axis_name=axis_name),

f_operator.defvjp(f_operator_fwd, f_operator_bwd)

@partial(jax.custom_vjp, nondiff_argnums=(1,))
def g_operator(x, axis_name):
    return jax.lax.psum(x, axis_name=axis_name)

def g_operator_fwd(x, axis_name):
    return g_operator(x, axis_name), ()

def g_operator_bwd(axis_name, res, g):
    return g,

g_operator.defvjp(g_operator_fwd, g_operator_bwd)


def test_mlp_model_parallel():
    lr = 0.1
    n_epoch = 1

    def loss_serial(x, y, w1, w2):
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    def step_serial(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_serial, argnums=(2, 3))(x, y, w1, w2)
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_serial(x, y, w1, w2):
        for i in range(n_epoch):
            w1, w2 = step_serial(x, y, w1, w2)
        return w1, w2

    def loss_parallel(x, y, w1, w2):
        x = f_operator(x, axis_name='model_parallel')
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        x = g_operator(x, axis_name='model_parallel')
        return ((x - y) ** 2).mean()

    @partial(jax.pmap, in_axes=(None, None, 1, 0), out_axes=(1, 0),
             axis_name='model_parallel')
    def step_parallel(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_parallel, argnums=(2, 3))(x, y, w1, w2)
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_parallel(x, y, w1, w2):
        model_parallel = len(jax.devices())

        w1 = split(w1, 1, model_parallel)
        w2 = split(w2, 0, model_parallel)

        for i in range(n_epoch):
            w1, w2 = step_parallel(x, y, w1, w2)

        return unsplit(w1, 1), unsplit(w2, 0)

    N = 8
    D = 128

    np.random.seed(0)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = train_serial(x, y, w1, w2)
    w1_parallel, w2_parallel = train_parallel(x, y, w1, w2)

    np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-4)
    np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-4)


def test_mlp_data_parallel():
    lr = 0.1
    n_epoch = 1

    def loss_serial(x, y, w1, w2):
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    def step_serial(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_serial, argnums=(2, 3))(x, y, w1, w2)
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_serial(x, y, w1, w2):
        for i in range(n_epoch):
            w1, w2 = step_serial(x, y, w1, w2)
        return w1, w2

    def loss_parallel(x, y, w1, w2):
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    @partial(jax.pmap, in_axes=(0, 0, None, None), out_axes=(None, None),
             axis_name='data_parallel')
    def step_parallel(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_parallel, argnums=(2, 3))(x, y, w1, w2)
        g_w1 = jax.lax.pmean(g_w1, axis_name='data_parallel')
        g_w2 = jax.lax.pmean(g_w2, axis_name='data_parallel')
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_parallel(x, y, w1, w2):
        data_parallel = len(jax.devices())

        x = split(x, 0, data_parallel)
        y = split(y, 0, data_parallel)

        for i in range(n_epoch):
            w1, w2 = step_parallel(x, y, w1, w2)

        return w1, w2

    N = 8
    D = 128

    np.random.seed(0)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = train_serial(x, y, w1, w2)
    w1_parallel, w2_parallel = train_parallel(x, y, w1, w2)

    np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-4)
    np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-4)


def test_mlp_data_model_parallel():
    lr = 0.1
    n_epoch = 1

    def loss_serial(x, y, w1, w2):
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    def step_serial(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_serial, argnums=(2, 3))(x, y, w1, w2)
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_serial(x, y, w1, w2):
        for i in range(n_epoch):
            w1, w2 = step_serial(x, y, w1, w2)
        return w1, w2

    def loss_parallel(x, y, w1, w2):
        x = f_operator(x, axis_name='model_parallel')
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        x = g_operator(x, axis_name='model_parallel')
        return ((x - y) ** 2).mean()

    @partial(jax.pmap, in_axes=(None, None, 1, 0), out_axes=(1, 0),
             axis_name='model_parallel')
    def step_model_parallel(x, y, w1, w2):
        g_w1, g_w2 = jax.grad(loss_parallel, argnums=(2, 3))(x, y, w1, w2)
        return g_w1, g_w2

    @partial(jax.pmap, in_axes=(0, 0, None, None), out_axes=(None, None),
             axis_name='data_parallel')
    def step_data_parallel(x, y, w1, w2):
        g_w1, g_w2 = step_model_parallel(x, y, w1, w2)
        g_w1 = jax.lax.pmean(g_w1, axis_name='data_parallel')
        g_w2 = jax.lax.pmean(g_w2, axis_name='data_parallel')
        return w1 - lr * g_w1, w2 - lr * g_w2

    def train_parallel(x, y, w1, w2):
        model_parallel = 2
        data_parallel = len(jax.devices()) // model_parallel

        x = split(x, 0, data_parallel)
        y = split(y, 0, data_parallel)
        w1 = split(w1, 1, model_parallel)
        w2 = split(w2, 0, model_parallel)

        for i in range(n_epoch):
            w1, w2 = step_data_parallel(x, y, w1, w2)

        return unsplit(w1, 1), unsplit(w2, 0)

    N = 8
    D = 128

    np.random.seed(0)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = train_serial(x, y, w1, w2)
    w1_parallel, w2_parallel = train_parallel(x, y, w1, w2)

    np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-4)
    np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-4)


if __name__ == "__main__":
    test_mlp_model_parallel()
    test_mlp_data_parallel()
    test_mlp_data_model_parallel()

