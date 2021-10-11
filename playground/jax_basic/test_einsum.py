import jax
import jax.numpy as jnp


@jax.jit
def func(a):
    c = jnp.einsum("GSE->GE", a)
    return c

a = jnp.ones((2, 4, 8))
func(a)
