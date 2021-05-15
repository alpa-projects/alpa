"""Test distributed mulit-host device mesh."""

from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from parax import parallelize, DeviceCluster, global_config, testing


class DeviceMeshTest(unittest.TestCase):
    def setUp(self):
        global_config.shard_parallel_strategy = "auto_sharding"
        self.device_cluster = DeviceCluster()

    def test_add_one(self):
        physical_mesh = self.device_cluster.get_physical_mesh()
        total_devices = len(physical_mesh.host_ids) * physical_mesh.num_devices_per_host
        logical_mesh = physical_mesh.get_logical_mesh([1, total_devices], [1, 1], [1, 1])

        @parallelize(devices=logical_mesh)
        def add_one(x):
            x = x + 1
            return x

        @parallelize(devices=logical_mesh)
        def multiply_two(x):
            x = x * 2
            return x

        a = jnp.ones((1000, 1000))
        out = add_one(a)
        out = multiply_two(out)

        np.testing.assert_allclose(out._value, (np.ones_like(a) + 1) * 2)

    def test_mlp(self):
        pass


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DeviceMeshTest('test_add_one'))
    #suite.addTest(DeviceMeshTest('test_mlp'))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())

