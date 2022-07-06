"""Test cross-mesh resharding."""
import unittest

import numpy as np
import ray

from alpa import init
from alpa.device_mesh import get_global_virtual_physical_mesh, next_array_uuids
from alpa.global_env import global_config


class XLANCCLTest(unittest.TestCase):

    def setUp(self):
        init(cluster="ray")

    @unittest.skip("manually calling allgather is deprecated")
    def test_xla_nccl_allgather(self):
        backup_nccl_mode = global_config.nccl_mode
        global_config.nccl_mode = "xla_extension"

        mesh_shape = (1, 4)
        size = (4, 4)
        virtual_mesh = get_global_virtual_physical_mesh()
        mesh = virtual_mesh.slice_2d(range(mesh_shape[0]),
                                     [range(mesh_shape[1])] *
                                     mesh_shape[0]).get_physical_mesh()
        worker = mesh.workers[0]
        device_ids = np.arange(mesh.num_devices_per_host)

        # Put buffers
        ary_uuid = next_array_uuids(1)[0]
        shard_len = size[0] // mesh.num_devices_per_host
        shards = []
        for i in range(mesh.num_devices_per_host):
            data = np.zeros(size, dtype=int)
            data[i * shard_len:(i + 1) * shard_len, :] = i
            shards.append(data)
        ray.get(worker.put_buffers.remote(ary_uuid, shards, 1, 0))

        # Put allgather task
        output_slice = [slice(0, size[0], None), slice(0, size[1], None)]
        tensor_slices = []
        for i in range(mesh.num_devices_per_host):
            tensor_slices.append([
                slice(i * shard_len, (i + 1) * shard_len, None),
                slice(0, size[1], None)
            ])
        ray.get(
            worker.put_resharding_allgather_task.remote(
                0, (ReshardingAllGatherSpec(device_ids, tensor_slices,
                                            output_slice),)))

        # Run allgather task
        ray.get(worker.run_allgather_task.remote(0, ary_uuid))
        refs = ray.get(worker.get_buffers.remote(ary_uuid))
        for i in range(4):
            for j in range(4):
                assert refs[i][j * shard_len, 0] == j

        global_config.nccl_mode = backup_nccl_mode


def suite():
    suite = unittest.TestSuite()
    suite.addTest(XLANCCLTest("test_xla_nccl_allgather"))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())
