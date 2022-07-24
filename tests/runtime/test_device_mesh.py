"""Test distributed mulit-host device mesh."""

import os
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import numpy as np
import ray

from alpa import init, parallelize, fetch, DistributedArray
from alpa.device_mesh import get_global_physical_mesh
from alpa.testing import assert_allclose


class DeviceMeshTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    def test_add_one(self):

        @parallelize
        def add_one(x):
            return x + 1

        @parallelize
        def multiply_two(x):
            return x * 2

        # Run computation
        a = jnp.ones((512, 512))
        out = add_one(a)
        out = multiply_two(out)

        # Check results
        assert_allclose(np.array(out), (np.ones_like(a) + 1) * 2)

    def test_distributed_array(self):
        physical_mesh = get_global_physical_mesh(create_if_not_exist=True)
        logical_mesh = physical_mesh.get_logical_mesh()

        array = jnp.arange(64).reshape([8, 8])
        sharding_spec = logical_mesh.make_tile_spec(array, [0, 1], [0, 1])
        indices = sharding_spec.indices(array.shape).flatten()
        dis_a = physical_mesh.shard_args_to_arrays([array.aval], [indices],
                                                   [sharding_spec], [array])[0]

        assert_allclose(array, dis_a)

    def test_preshard_args(self):

        @parallelize
        def add_one(x):
            return x + 1

        a = jnp.ones((64, 64))
        a, = add_one.preshard_dynamic_args(a)
        assert isinstance(a, DistributedArray)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DeviceMeshTest("test_add_one"))
    suite.addTest(DeviceMeshTest("test_distributed_array"))
    suite.addTest(DeviceMeshTest("test_preshard_args"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
