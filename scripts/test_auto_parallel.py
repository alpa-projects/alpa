from functools import partial

import numpy as np
import jax
import jax.numpy as jnp


def auto_parallel(func, batch, weights):
    n_devices = len(jax.devices())

    def process_batch(batch):
        def split(x):
            assert x.shape[0] % n_devices == 0
            return x.reshape((n_devices, x.shape[0] // n_devices) + x.shape[1:])
        return jax.tree_util.tree_map(split, batch)

    def process_weight(weight):
        return weight

    jaxpr = jax.make_jaxpr(func)(batch, weights)

    func_parallel = jax.pmap(func, axis_name='auto_parallel_batch',
        in_axes=(0, None), out_axes=None)

    return func_parallel, process_batch, process_weight


def annotate_gradient(gradients):
    from jax.core import thread_local_state

    axis_env = thread_local_state.trace_state.axis_env
    in_auto_parallel = False
    for x in axis_env:
        if x.name == 'auto_parallel_batch':
            in_auto_parallel = True

    if in_auto_parallel:
        return jax.lax.pmean(gradients, 'auto_parallel_batch')
    else:
        return gradients


def test_auto_parallel():
    lr = 0.1
    n_epoch = 1

    def loss_serial(batch, weights):
        x, y = batch
        w1, w2 = weights
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    def step_serial(batch, weights):
        gradients = jax.grad(loss_serial, argnums=1)(batch, weights)
        gradients = annotate_gradient(gradients)
        return [w - g * lr for w, g in zip(weights, gradients)]

    def train_serial(batch, weights):
        for i in range(n_epoch):
            weights = step_serial(batch, weights)
        return weights

    def train_parallel(batch, weights):
        step_parallel, process_batch, process_weight =\
            auto_parallel(step_serial, batch, weights)

        weights = process_weight(weights)

        for i in range(n_epoch):
            pbatch = process_batch(batch)
            weights = step_parallel(pbatch, weights)

        return weights

    N = 8
    D = 128

    np.random.seed(0)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = train_serial((x, y), (w1, w2))
    w1_parallel, w2_parallel = train_parallel((x, y), (w1, w2))

    np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-4)
    np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-4)


if __name__ == "__main__":
   test_auto_parallel()

