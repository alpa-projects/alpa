import numpy as np
import jax
from jax import numpy as jnp

def test_jit_cache():

    @jax.jit
    def add_one(x):
        # return jnp.arange(x[0])
        return x + 1

    a = jnp.ones(10)

    print(add_one(a))
    print(add_one(a))
    print(add_one(a))


def test_non_jit():
    a = jnp.array(np.ones(10))
    b = jnp.array(np.ones(10))
    c = a + b
    c = a + c
    c = a + c

    print(c)


if __name__ == "__main__":
    #test_jit_cache()
    test_non_jit()

