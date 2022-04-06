from typing import Any, List, Sequence
import ray

from alpa.global_env import global_config
from alpa.device_mesh import DistributedPhysicalDeviceMesh
from alpa.pipeline_parallel.cross_mesh_resharding import CollectiveGroup
from alpa.util import OrderedSet


class DistributedPhysicalDeviceMeshGroup:
    """A list of physical devices that forms a pipeline."""

    def __init__(self, meshes: Sequence[DistributedPhysicalDeviceMesh]):
        self.meshes = list(meshes)
        self.collective_groups: List[List[Any]] = [
            [None for _ in range(len(self))] for _ in range(len(self))
        ]

    def __getitem__(self, index):
        return self.meshes[index]

    def __len__(self):
        return len(self.meshes)

    def index(self, *args, **kwargs):
        return self.meshes.index(*args, **kwargs)

    def establish_nccl_group(self, src_mesh_id: int, dst_mesh_id: int):
        """Establish NCCL group between two meshes."""
        assert src_mesh_id < dst_mesh_id
        if self.collective_groups[src_mesh_id][dst_mesh_id] is not None:
            # Already established
            return
        src_mesh = self.meshes[src_mesh_id]
        dst_mesh = self.meshes[dst_mesh_id]
        device_strs = OrderedSet(src_mesh.device_strs + dst_mesh.device_strs)
        cg = CollectiveGroup(device_strs, src_mesh, dst_mesh)
        if global_config.eagerly_create_communicators:
            cg.instantiate_now()
        else:
            cg.instantiate()
        self.collective_groups[src_mesh_id][dst_mesh_id] = cg
        self.collective_groups[dst_mesh_id][src_mesh_id] = cg

    def destroy_collective_groups(self):
        for i in range(len(self)):
            for j in range(len(self)):
                if i < j and self.collective_groups[i][j] is not None:
                    self.collective_groups[i][j].destroy()

    def shutdown(self):
        self.destroy_collective_groups()
        for mesh in self.meshes:
            mesh.shutdown()

    def exception_shutdown(self):
        """In this shutdown, some actors might have died."""
        # recycle collective group info
        for i in range(len(self)):
            for j in range(len(self)):
                if i < j and self.collective_groups[i][j]:
                    group_name = self.collective_groups[i][j].group_name
                    # TODO(Hao): move this part of recycling to ray.util.collective instead of here.
                    name = "info_" + group_name
                    try:
                        store = ray.get_actor(name)
                        ray.kill(store)
                    except ValueError:
                        pass
        # TODO(Hao): recycle the NCCLUniqueID named actor. Their name is MD5 hashed.
        #            each of them will takes 1 CPU.
        # recycle info actors
        for mesh in self.meshes:
            mesh.shutdown(forced=True)
