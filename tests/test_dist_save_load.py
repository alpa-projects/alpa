"""Test distributed mulit-host device mesh."""

import subprocess
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import ray

from alpa import DeviceCluster, DistributedArray
from alpa.testing import assert_allclose


class DistSaveLoadTest(unittest.TestCase):

    def setUp(self):
        ray.init(address="auto", ignore_reinit_error=True)

    def tearDown(self):
        ray.shutdown()

    def test_distributed_array_save_load(self):
        # Launch a device mesh contains four devices
        device_cluster = DeviceCluster()
        if device_cluster.num_devices < 4:
            self.skipTest(
                "This unit test requires a cluster with at least 4 devices! ")
        host_num = min(len(device_cluster.host_info), 4)
        device_per_host = 4 // host_num
        physical_mesh = device_cluster.get_physical_mesh(
            list(range(host_num)), device_per_host)
        logical_mesh = physical_mesh.get_logical_mesh([2, 2])

        global_input_shape = (4, 2)
        num = np.prod(np.array(global_input_shape))

        # Build DistributedArray to be saved
        # [[0,1],          [[0],  [[1],
        #  [2,3],  shard    [2]]   [3]]
        #  [4,5],  ====>   [[4],  [[5],
        #  [6,7]]           [6]]   [7]]
        global_input_data1 = jnp.arange(num).reshape(global_input_shape)
        input_sharding_spec = logical_mesh.make_tile_spec(
            global_input_data1, [0, 1], [0, 1])
        input_indices = input_sharding_spec.indices(
            global_input_data1.shape).flatten()
        (dist_input_data1,) = physical_mesh.shard_args_to_arrays(
            (jax.ShapedArray(global_input_data1.shape, jnp.int32),),
            (input_indices,), (input_sharding_spec,), (global_input_data1,))

        # Check the DistributedArray's remote buffers
        remote_buffers1 = dist_input_data1.device_mesh.get_remote_buffers(
            dist_input_data1.remote_buffers, batching=True)
        desired_buffers1 = [[[0], [2]], [[1], [3]], [[4], [6]], [[5], [7]]]
        assert_allclose(np.array(desired_buffers1), np.array(remote_buffers1))

        # Save the DistributedArray (one replica only)
        tmpdir = tempfile.TemporaryDirectory()
        subprocess.run(["rm", "-rf", tmpdir.name])
        dist_input_data1.save(tmpdir.name)

        # Load previously saved DistributedArray with a different shardingSpec
        # [[0,1],          [[0,1],  [[0,1],
        #  [2,3],  shard    [2,3]]   [2,3]]
        #  [4,5],  ====>   [[4,5],  [[4,5],
        #  [6,7]]           [6,7]]   [6,7]]
        load_sharding_spec = logical_mesh.make_tile_spec(
            global_input_data1, [0, 1], [0])
        dist_load_data1 = DistributedArray.load(
            tmpdir.name, jax.ShapedArray(global_input_data1.shape, jnp.int32),
            physical_mesh, load_sharding_spec)

        # Check the DistributedArray's remote buffers
        remote_buffers2 = dist_load_data1.device_mesh.get_remote_buffers(
            dist_load_data1.remote_buffers, batching=True)
        desired_buffers2 = [[[0, 1], [2, 3]], [[0, 1], [2, 3]], [[4, 5], [6,
                                                                          7]],
                            [[4, 5], [6, 7]]]
        assert_allclose(np.array(desired_buffers2), np.array(remote_buffers2))

        # Cleanup
        physical_mesh.shutdown()


def suite():
    suite = unittest.TestSuite()
    suite.addTest(DistSaveLoadTest("test_distributed_array_save_load"))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
