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


def input_iter_func(start, end, batch_size):
    num_batches = (end - start) // batch_size
    for i in range(num_batches):
        yield (i * np.ones((batch_size, 32), dtype=np.float32), i * np.ones(
            (batch_size,), dtype=np.int32))


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def run_test(self, sharding_specs):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        num_devices = physical_mesh.num_devices

        batch_size = 64
        num_samples = 256
        avals = [
            jax.core.ShapedArray((batch_size, 32), jnp.float32),
            jax.core.ShapedArray((batch_size,), jnp.int32)
        ]
        prefetch_size = 2

        data_loader = MeshDriverDataLoader(batch_size, num_samples,
                                           input_iter_func, avals,
                                           sharding_specs, physical_mesh,
                                           prefetch_size)
        expected_data_loader = input_iter_func(0, num_samples, batch_size)

        for actual_batch, expected_batch in zip(data_loader,
                                                expected_data_loader):
            assert_allclose(actual_batch, expected_batch)

    def test_data_parallel(self):
        num_devices = DeviceCluster().num_devices

        sharding_specs = [
            pxla.ShardingSpec((pxla.Chunked((num_devices,)), pxla.NoSharding()),
                              (pxla.ShardedAxis(0),)),
            pxla.ShardingSpec((pxla.Chunked((num_devices,)),),
                              (pxla.ShardedAxis(0),))
        ]
        self.run_test(sharding_specs)

    def test_model_parallel(self):
        num_devices = DeviceCluster().num_devices

        sharding_specs = [
            pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked((num_devices,))),
                              (pxla.ShardedAxis(0),)),
            pxla.ShardingSpec((pxla.NoSharding(),),
                              (pxla.Replicated(num_devices),))
        ]
        self.run_test(sharding_specs)

    def test_data_model_parallel(self):
        dp = 2
        mp = DeviceCluster().num_devices // dp
        sharding_specs = [
            pxla.ShardingSpec((pxla.Chunked((dp,)), pxla.Chunked((mp,))),
                              (pxla.ShardedAxis(0), pxla.ShardedAxis(1))),
            pxla.ShardingSpec((pxla.Chunked((dp,)),), (
                pxla.ShardedAxis(0),
                pxla.Replicated(mp),
            ))
        ]
        self.run_test(sharding_specs)


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DataLoaderTest("test_data_parallel"))
    suite.addTest(DataLoaderTest("test_model_parallel"))
    suite.addTest(DataLoaderTest("test_data_model_parallel"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
