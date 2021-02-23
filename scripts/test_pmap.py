import jax
import jax.numpy as jnp

jaxpr = jax.make_jaxpr(jax.pmap(lambda x: x ** 2))(jnp.arange(1))

f = lambda x: x / jax.lax.psum(x, axis_name='i')
out = jax.pmap(f, axis_name='i')(jnp.ones((1, 4)))

