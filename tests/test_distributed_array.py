"""Test distributed array."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp

from parax import parallelize, DeviceCluster, global_config, testing


class DistributedArrayTest(unittest.TestCase):
    def setUp(self):
        global_config.shard_parallel_strategy = "auto_sharding"
        self.device_cluster = DeviceCluster()

    def test_distributed_array(self):
        physical_mesh = self.device_cluster.get_physical_mesh()
        physical_mesh.launch_distributed_xla_service()
        physical_mesh.sync_workers()

        logical_mesh = physical_mesh.get_default_logical_mesh()

        array = jnp.ones((12, 12))
        sharding_spec = logical_mesh.make_tile_spec(array, [0, 1], [0, 1])
        indices = sharding_spec.indices(array.shape).flatten()

        remote_a = physical_mesh._shard_args([indices], (array,))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DistributedArrayTest('test_distributed_array'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

