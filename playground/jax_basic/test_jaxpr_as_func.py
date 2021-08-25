import jax
from jax.core import jaxpr_as_fun
import jax.numpy as jnp

def func(a, b):
    return a + b

N = 2 ** 10

a = jnp.ones((N, N))
b = jnp.ones((N, N))

jaxpr = jax.make_jaxpr(func)(a, b)
print(type(jaxpr))

func = jaxpr_as_fun(jaxpr)

a = jnp.ones((2*N, N))
b = jnp.ones((2*N, N))
print(func(a, b))

