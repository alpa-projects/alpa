import jax, jax.numpy as jnp

def func(a, b):
    c =  jnp.asarray(a, jnp.int32) @ jnp.asarray(b, jnp.int32)
    #c = a @ b
    c = c.transpose()
    c += a
    return c

a = jnp.ones((100, 100))
b = jnp.ones((100, 100))

m = jax.xla_computation(func)(a, b).as_hlo_module()
print(m.to_string())
r = jax.lib.xla_client._xla.hlo_module_count_flop_dot_conv_only(m)
print(r)

