from functools import partial 
import jax
from jax import lax
import jax.numpy as jnp


def debug_pmap():
    @jax.pmap
    def func(x, w):
        return x @ w

    y = func(jnp.ones((2, 4)), jnp.ones((2, 4)))
    print(y, type(y))


def test_nested_pmap():
    @partial(jax.pmap, axis_name='a0', in_axes=(0, None), out_axes=0)
    def add(a, b):
        # a.shape = (32, 64)
        # b.shape = (64, 2, 32)
        @partial(jax.pmap, axis_name='a1', in_axes=(None, 1), out_axes=1)
        def add_inner(x, y):
            # x.shape = (32, 64)
            # y.shape = (64, 32)
            return x @ y

        # ret.shape = (32, 2, 32)
        ret = add_inner(a, b)
        return ret

    a = jnp.ones((2, 32, 64))
    b = jnp.ones((64, 2, 32))

    #jaxpr = jax.make_jaxpr(add)(a, b)
    #print(jaxpr)
    #print(jaxpr.jaxpr.outvars[0].aval.shape)

    c = add(a, b)
    print(c)


def test_allreduce_sum():
    @partial(jax.pmap, axis_name='i')
    def normalize(x):
        return x / lax.psum(x, 'i')

    print(normalize(jnp.arange(2)))


if __name__ == "__main__":
    #debug_pmap()
    #test_nested_pmap()

    test_allreduce_sum()

