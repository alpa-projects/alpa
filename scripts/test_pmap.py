from functools import partial 
import jax
import jax.numpy as jnp

#f = lambda x: jax.lax.psum(x, axis_name='batch')
#y = jax.pmap(f, axis_name='batch')(jnp.ones((4, 4)))
#print(y, type(y))

@jax.pmap
def func(x, w):
    print("pmap A")
    return x @ w

print("pmap B")
y = func(jnp.ones((1, 4)), jnp.ones((1, 4)))
print("pmap C")
print(y, type(y))

#print(jax.make_jaxpr(func)(jnp.ones((4, 4)), jnp.ones((4, 4))))

