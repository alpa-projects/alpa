"""Test profiling of communication and compute costs."""

import os
import unittest

import jax.numpy as jnp
import ray

from parax import DeviceCluster, parallelize, set_parallelize_options


class ProfilingTest(unittest.TestCase):

    def setUp(self):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        ray.init(address="auto", ignore_reinit_error=True)

    def test_profile_allreduce(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        physical_mesh.profile_collective("all-reduce",
                                         size_range=range(18, 19),
                                         verbose=False)
        physical_mesh.shutdown()

    def test_profile_allgather(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        physical_mesh.profile_collective("all-gather",
                                         size_range=range(18, 19),
                                         verbose=False)
        physical_mesh.shutdown()

    def test_loading_profiling_result(self):
        device_cluster = DeviceCluster()
        physical_mesh = device_cluster.get_physical_mesh()
        physical_mesh.prof_result.record_all_reduce(((0, 1, 2, 3),), 1 << 10,
                                                    "float32", 0.2)
        physical_mesh.prof_result.record_all_reduce(((0, 1, 2, 3),), 1 << 11,
                                                    "float32", 0.4)
        physical_mesh.prof_result.record_all_reduce((
            (0, 1),
            (2, 3),
        ), 1 << 10, "float32", 0.2)
        physical_mesh.prof_result.record_all_reduce((
            (0, 1),
            (2, 3),
        ), 1 << 11, "float32", 0.4)
        set_parallelize_options(devices=physical_mesh)

        @parallelize
        def add_one(x):
            return x + 1

        add_one(jnp.ones(16))

        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(ProfilingTest("test_profile_allreduce"))
    suite.addTest(ProfilingTest("test_profile_allgather"))
    suite.addTest(ProfilingTest("test_loading_profiling_result"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
