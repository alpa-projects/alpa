import jax
from jax.core import jaxpr_as_fun
import jax.numpy as jnp

def func(a, b):
    return a + b

N = 2 ** 10

a = jnp.ones((N, N))
b = jnp.ones((N, N))

jaxpr = jax.make_jaxpr(func)(a, b)

func = jaxpr_as_fun(jaxpr)

print(func(a, b))

