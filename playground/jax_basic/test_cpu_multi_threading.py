
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')


@jax.jit
def add(a, b):
    for i in range(100):
        a = (a * b) + a
    return a


a = jnp.ones(1 << 30)
b = jnp.ones(1 << 30)
add(a, b)

