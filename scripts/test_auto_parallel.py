from functools import partial, wraps

import numpy as np
import jax
import jax.numpy as jnp

def test_auto_parallel():
    lr = 0.1
    n_epoch = 2

    def loss_serial(weights, batch):
        x, y = batch
        w1, w2 = weights
        x = x @ w1
        x = jax.nn.relu(x)
        x = x @ w2
        return ((x - y) ** 2).mean()

    def step_serial(weights, batch):
        gradients = jax.grad(loss_serial, argnums=0)(weights, batch)
        gradients = annotate_gradient(gradients)
        return [w - g * lr for w, g in zip(weights, gradients)]

    def train_serial(weights, batch):
        for i in range(n_epoch):
            weights = step_serial(weights, batch)
        return weights

    step_parallel = auto_parallel(step_serial)

    def train_parallel(weights, batch):
        for i in range(n_epoch):
            weights = step_parallel(weights, batch)

        return weights

    N = 8
    D = 64

    np.random.seed(0)
    x = np.random.uniform(size=(N, D))
    y = np.random.uniform(size=(N, D))
    w1 = np.random.uniform(size=(D, D))
    w2 = np.random.uniform(size=(D, D))

    w1_serial, w2_serial = train_serial((w1, w2), (x, y))
    w1_parallel, w2_parallel = train_parallel((w1, w2), (x, y))

    np.testing.assert_allclose(w1_serial, w1_parallel, rtol=1e-4)
    np.testing.assert_allclose(w2_serial, w2_parallel, rtol=1e-4)


if __name__ == "__main__":
   test_auto_parallel()

