"""Test distributed array."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp
import ray

from parax import parallelize, DeviceCluster, global_config, testing


class DistributedArrayTest(unittest.TestCase):
    def setUp(self):
        ray.init(address="auto")

    def test_distributed_array(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        logical_mesh = physical_mesh.get_default_logical_mesh()

        array = jnp.ones((16, 16))
        sharding_spec = logical_mesh.make_tile_spec(array, [0, 1], [0, 1])
        indices = sharding_spec.indices(array.shape).flatten()
        remote_a = physical_mesh._shard_args([indices], (False,), (array,))
        physical_mesh.shutdown()
        ray.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DistributedArrayTest("test_distributed_array"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

