import jax
from jax import numpy as jnp


@jax.pmap
def many_args(*args):
    x = 0
    for i in range(len(args)):
        x += args[i]
    return x

N = 110

args = [
  jnp.ones((4, 10)) for _ in range(N)
]

out = many_args(*args)
print(out)

