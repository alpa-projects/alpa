"""Test profiling of communication and compute costs."""

from functools import partial
import os
import pickle
import unittest

from flax import linen as nn
from flax import optim
import jax
import jax.numpy as jnp
import numpy as np
import ray

from parax import parallelize, DeviceCluster, global_config, testing
from parax.testing import assert_allclose


class ProfilingTest(unittest.TestCase):
    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        ray.init(address="auto", ignore_reinit_error=True)

    def test_profile_allreduce(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        physical_mesh.profile_collective(
            "all-reduce", size_range=range(18, 19), verbose=False)
        physical_mesh.shutdown()

    def test_profile_allgather(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        physical_mesh.profile_collective(
            "all-gather", size_range=range(18, 19), verbose=False)
        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ProfilingTest("test_profile_allreduce"))
    suite.addTest(ProfilingTest("test_profile_allgather"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())

