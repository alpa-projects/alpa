import jax
import jax.numpy as jnp

N = 2 ** 20

def func(value):
    if jax.cond(value):
        a = jnp.ones((2, N))
    else:
        a = jnp.ones((1, N))

    c = a * 2
    return c

jaxpr = jax.make_jaxpr(func, static_argnums=[0])(True)

print(jaxpr.out_avals)

