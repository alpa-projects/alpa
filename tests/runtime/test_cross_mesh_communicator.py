import unittest

import ray
from alpa import init
from alpa.device_mesh import create_and_record_cross_mesh_collective_communicators, get_global_cluster
from alpa.pipeline_parallel.stage_construction import get_sliced_virtual_submeshes


class CrossMeshCollectiveCommunicatorTest(unittest.TestCase):

    def setUp(self) -> None:
        init("ray")

    def test_create_and_set(self):
        virtual_mesh = get_global_cluster().get_virtual_physical_mesh(
            host_ids=[0], num_devices_per_host=4)
        submesh_shapes = [(1, 2)] * 2
        sliced_virtual_meshes = get_sliced_virtual_submeshes(
            virtual_mesh, submesh_shapes)
        virtual_mesh.get_physical_mesh_group(sliced_virtual_meshes)
        mesh_group = virtual_mesh.launched_physical_mesh_group
        meshes = mesh_group.meshes
        ray.get(create_and_record_cross_mesh_collective_communicators(meshes))


def suite():
    suite = unittest.TestSuite()
    suite.addTest(CrossMeshCollectiveCommunicatorTest("test_create_and_set"))

    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    runner.run(suite())
