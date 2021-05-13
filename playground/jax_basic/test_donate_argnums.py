import jax
import jax.numpy as jnp

def test_donate_argnums():

    def func(a, b, tile_factor):
        return jnp.tile(a, tile_factor) + jnp.tile(b, tile_factor)


    donated_func = jax.jit(func, donate_argnums=(0, 1), static_argnums=(2,))

    a = jnp.ones(10)
    b = jnp.ones(10)
    c = donated_func(a, b, 1)
    try:
        print(a)
    except RuntimeError:
        print("a is donated and deleted")

    a = jnp.ones(10)
    b = jnp.ones(10)
    c = donated_func(a, b, 2)
    try:
        print(a)
        print("a is not donated")
    except RuntimeError:
        print("a is donated and deleted")


if __name__ == "__main__":
    test_donate_argnums()

