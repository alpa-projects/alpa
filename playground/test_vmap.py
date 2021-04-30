import jax
import jax.numpy as jnp

vv = lambda x, y: jnp.vdot(x, y)

v = jnp.ones(10)
m = jnp.ones((10, 10))

print(jax.make_jaxpr(vv)(v, v))

mv = jax.vmap(vv, (0, None), 0)
print(jax.make_jaxpr(mv)(m, v))


