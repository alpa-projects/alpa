import numpy as np
import jax
from jax import numpy as jnp

def test_jit_cache():

    @jax.jit
    def add_one(x):
        return x + 1

    a = jnp.ones(10)

    print(add_one(a))
    print(add_one(a))
    print(add_one(a))


def test_cache_closure():
    outer_scope = [0]

    @jax.jit
    def add_one(x):
        print('call add_one')
        return x + outer_scope[0]

    a = jnp.ones(10)

    print(add_one(a))
    print(add_one(a))
    outer_scope[0] = 1
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
    test_cache_closure()
    #test_non_jit()

