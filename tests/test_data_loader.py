"""Test distributed mesh data loader."""

import os
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import numpy as np
import ray

from alpa import DeviceCluster, MeshDriverDataLoader
from alpa.testing import assert_allclose


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_data_loader_data_parallel(self):

        device_cluster = DeviceCluster()

        physical_mesh = device_cluster.get_physical_mesh()
        num_devices = physical_mesh.num_devices

        def input_iter_func(start, end, batch_size):
            for i in range(start, end):
                yield (i * np.ones((batch_size, 16), dtype=np.float32),
                       i * np.ones((batch_size,), dtype=np.int32))

        batch_size = 32
        num_samples = 128
        avals = [jax.core.ShapedArray((batch_size, 16), jnp.float32),
                 jax.core.ShapedArray((batch_size,), jnp.int32)]
        sharding_specs = [
            pxla.ShardingSpec(
                (pxla.Chunked((num_devices,)), pxla.NoSharding()),
                (pxla.ShardedAxis(0),)
            ),
            pxla.ShardingSpec(
                (pxla.Chunked((num_devices,)),),
                (pxla.ShardedAxis(0),)
            )
        ]
        prefetch_size = 1

        data_loader = MeshDriverDataLoader(batch_size, num_samples, input_iter_func,
            avals, sharding_specs, physical_mesh, prefetch_size)

        expected_data_loader = input_iter_func(0, num_samples, batch_size)

        for actual_batch, expected_batch in zip(data_loader, expected_data_loader):
            assert_allclose(actual_batch, expected_batch)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DataLoaderTest("test_data_loader_data_parallel"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

