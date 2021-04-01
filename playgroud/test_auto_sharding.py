import numpy as np

import jax
import jax.numpy as jnp

from paranum import parallelize


def test_matmul():

    @parallelize
    def matmul(a, b):
        return a @ b

    x = jnp.ones((128, 128))
    y = jnp.ones((128, 128))

    c = matmul(x, y)
    c.block_until_ready()

    np.testing.assert_allclose(np.array(c), np.array(x @ y))


if __name__ == "__main__":
    test_matmul()

