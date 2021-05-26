import time

import jax
import jax.numpy as jnp


def test_synchronize_all_activity():
    a = jnp.ones((8192, 8192))
    b = jnp.ones((8192, 8192))
    device = jax.devices()[0]

    @jax.jit
    def func(a, b):
        return a @ b

    tic = time.time()
    for i in range(100):
        b = func(b, a)

 
    device.synchronize_all_activity()
    print(time.time() - tic)
    b.block_until_ready()
    print(time.time() - tic)


if __name__ == "__main__":
    test_synchronize_all_activity()
