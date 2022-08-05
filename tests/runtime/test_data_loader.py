"""Test distributed mesh data loader."""
import os
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
from jax.interpreters import pxla
import numpy as np

from alpa import init, MeshDriverDataLoader
from alpa.parallel_plan import PlacementSpec
from alpa.device_mesh import get_global_physical_mesh
from alpa.testing import assert_allclose
from alpa.testing import data_loader_input_iter_func as input_iter_func


class DataLoaderTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")
        self.physical_mesh = get_global_physical_mesh(create_if_not_exist=True)

    def run_test(self, sharding_specs):
        batch_size = 64
        num_samples = 256
        feature_dim = 32
        avals = [
            jax.core.ShapedArray((batch_size, feature_dim), jnp.float32),
            jax.core.ShapedArray((batch_size,), jnp.int32)
        ]
        placement_specs = [
            PlacementSpec(aval, (self.physical_mesh.mesh_id,), (sharding_spec,))
            for aval, sharding_spec in zip(avals, sharding_specs)
        ]
        prefetch_size = 2

        data_loader = MeshDriverDataLoader(batch_size, num_samples,
                                           input_iter_func, placement_specs,
                                           prefetch_size)
        expected_data_loader = input_iter_func(0, num_samples, batch_size)

        actual_x = []
        actual_y = []
        expected_x = []
        expected_y = []
        for actual_batch, expected_batch in zip(data_loader,
                                                expected_data_loader):
            actual_x.append(np.array(actual_batch[0]))
            actual_y.append(np.array(actual_batch[1]))
            expected_x.append(np.array(expected_batch[0]))
            expected_y.append(np.array(expected_batch[1]))

        actual_x = np.concatenate(actual_x)
        actual_y = np.concatenate(actual_y)
        expected_x = np.concatenate(expected_x)
        expected_y = np.concatenate(expected_y)

        # Check that actual_x is a permutation of expected_x.
        for i in range(feature_dim):
            assert np.sum(actual_x[:, i]) == np.sum(expected_x[:, i])
        # Check that actual_y is a permutation of expected_y.
        assert np.sum(actual_y) == np.sum(expected_y)

    def test_data_parallel(self):
        num_devices = self.physical_mesh.num_devices

        sharding_specs = [
            pxla.ShardingSpec((pxla.Chunked((num_devices,)), pxla.NoSharding()),
                              (pxla.ShardedAxis(0),)),
            pxla.ShardingSpec((pxla.Chunked((num_devices,)),),
                              (pxla.ShardedAxis(0),))
        ]
        self.run_test(sharding_specs)

    def test_model_parallel(self):
        num_devices = self.physical_mesh.num_devices

        sharding_specs = [
            pxla.ShardingSpec((pxla.NoSharding(), pxla.Chunked((num_devices,))),
                              (pxla.ShardedAxis(0),)),
            pxla.ShardingSpec((pxla.NoSharding(),),
                              (pxla.Replicated(num_devices),))
        ]
        self.run_test(sharding_specs)

    def test_data_model_parallel(self):
        dp = 2
        mp = self.physical_mesh.num_devices // dp
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
