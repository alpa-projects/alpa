from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
from jax.experimental import PartitionSpec as P
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit, with_sharding_constraint


def test_basic1d():
    @partial(pjit,
             in_axis_resources=(P('x'), P('x')),
             out_axis_resources=None)
    def f(x, y):
      return x + y

    x = np.ones((8, 8, 2))

    mesh_devices = np.array(jax.devices()[:1])
    with mesh(mesh_devices, ('x',)):
        actual = f(x, x + 1)


if __name__ == "__main__":
    test_basic1d()

