
import jax
import jax.numpy as jnp

def test_attention_slice():

    @jax.jit
    def func(a, b):
        a1, a2 = jnp.split(a, 2)

        out = a1 @ b + a2 @ b

    a = jnp.ones((2, 100, 100))
    b = jnp.ones((100, 100))

    jaxpr = jax.make_jaxpr(func)(a, b)
    print(jaxpr)

    #out = func(a, b)

if __name__ == "__main__":
    test_attention_slice()

