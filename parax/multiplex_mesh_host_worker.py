import os
import numpy as np
from typing import List, Union, Tuple


class MultiplexMeshHostWorker:
    """
    A thin wrapper around the mesh host worker to override ray's num_gpus
    option to allow for GPU multiplexing.
    """
    def __init__(self, server_address, num_hosts, host_id, devices):
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            map(str, devices))
        from parax.device_mesh import MeshHostWorker
        self.worker = MeshHostWorker(server_address, num_hosts, host_id)

    ##### Buffer Related Functions #####
    def put_buffer(self, uuid: int, device_id: int, data: np.ndarray):
        return self.worker.put_buffer(uuid, device_id, data)

    def put_non_zero_buffer(self,
                            uuid: int,
                            device_id: int,
                            shape: Tuple[int, ...],
                            dtype=np.float32):
        return self.worker.put_non_zero_buffer(uuid, device_id, shape, dtype)

    def get_buffers(self, uuids: Union[List[int], int]):
        return self.worker.get_buffers(uuids)

    def delete_buffers(self, uuids: Union[List[int], int]):
        return self.worker.delete_buffers(uuids)

    def block_until_ready_buffers(self, uuids: Union[List[int], int]):
        return self.worker.block_until_ready_buffers(uuids)

    ##### Executable Related Functions #####
    def put_executable(self, uuid: int, executable_class, *args):
        return self.worker.put_executable(uuid, executable_class, *args)

    def delete_executable(self, uuid: int):
        return self.worker.delete_executable(uuid)

    def run_executable(self, uuid: int, *args, **kwargs):
        return self.worker.run_executable(uuid, *args, **kwargs)

    def get_exec_total_allocation_size(self, uuid: int):
        return self.worker.get_exec_total_allocation_size(uuid)

    ##### Cross Mesh Resharding Related Functions #####
    def send_tile(self, uuid, offset, dst_rank, dst_gpu_idx, group_name):
        return self.worker.send_tile(uuid, offset, dst_rank, dst_gpu_idx, group_name)

    def recv_tile(self, uuid, device_id, indices_in_dst_tile, src_rank,
                  src_gpu_idx, group_name):
        return self.worker.recv_tile(uuid, device_id, indices_in_dst_tile,
                                     src_rank, src_gpu_idx, group_name)

    def put_resharding_send_task(self, uuid, tasks, group_name):
        return self.worker.put_resharding_send_task(uuid, tasks, group_name)

    def put_resharding_recv_task(self, uuid, tasks, group_name):
        return self.worker.put_resharding_recv_task(uuid, tasks, group_name)

    def run_resharding_send_task(self, uuid, buf_uuids):
        return self.worker.run_resharding_send_task(uuid, buf_uuids)

    def run_resharding_recv_task(self, uuid, buf_uuids,
                                 set_empty_buffer=True):
        return self.worker.run_resharding_recv_task(uuid, buf_uuids,
                                                    set_empty_buffer)

    ##### Profiling Related Functions #####
    def profile_collective(self, primitive_name, size_range, replica_groups,
                           number, verbose):
        return self.worker.profile_collective(primitive_name, size_range,
                                              replica_groups, number, verbose)

    def profile_executable_with_dummy_inputs(self, uuid: int, **kwargs):
        return self.worker.profile_executable_with_dummy_inputs(uuid, **kwargs)

    # TODO(yonghao): the sync function should be carefully reconsidered
    def profile_resharding_send_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        return self.worker.profile_resharding_send_task(uuid, buf_uuids,
                                                        warmup, repeat, number,
                                                        sync)

    def profile_resharding_recv_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        return self.worker.profile_resharding_recv_task(uuid, buf_uuids,
                                                        warmup, repeat, number,
                                                        sync)

    def get_timer(self, name: str):
        return self.worker.get_timer(name)

    def reset_timer(self, name: str):
        return self.worker.reset_timer(name)

    ##### Other Functions #####
    def sync(self):
        return self.worker.sync()

    def shutdown(self):
        return self.worker.shutdown()

    def destroy_collective_group(self, group_name: str = "default"):
        return self.worker.destroy_collective_group(group_name)
