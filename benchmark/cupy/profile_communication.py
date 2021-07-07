import time

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray

MB = 1 << 20
GB = 1 << 30


def do_all_reduce(comm, in_buffer, out_buffer, size):
    comm.allReduce(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        0,
        cp.cuda.Stream.null.ptr,
    )


def do_all_gather(comm, in_buffer, out_buffer, size):
    comm.allGather(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        cp.cuda.Stream.null.ptr,
    )


@ray.remote(num_gpus=1)
class GpuHost:
    def __init__(self, global_id, nccl_uuid_list):
        self.global_id = global_id
        self.nccl_uuid_list = nccl_uuid_list
        self.ct = 0

    def init_communicator(self, groups):
        comm = None
        for group in groups:
            nccl_uuid = self.nccl_uuid_list[self.ct]
            self.ct += 1
            for device_id in group:
                if self.global_id == device_id:
                    assert comm is None
                    comm = cp.cuda.nccl.NcclCommunicator(
                        len(group), nccl_uuid, group.index(self.global_id))

        cp.cuda.Device(0).synchronize()
        return comm

    def profile_allreduce(self, size, dtype, groups):
        comm = self.init_communicator(groups)

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_reduce(comm, in_buffer, out_buffer, size)
        do_all_reduce(comm, in_buffer, out_buffer, size)

        number = min(max(15, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_reduce(comm, in_buffer, out_buffer, size)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.global_id == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = 2 * array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllReduce: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")

    def profile_allgather(self, size, dtype, groups):
        comm = self.init_communicator(groups)

        in_buffer = cp.ones(int(size) // len(groups[0]), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_gather(comm, in_buffer, out_buffer, size)
        do_all_gather(comm, in_buffer, out_buffer, size)

        number = min(max(15, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_gather(comm, in_buffer, out_buffer, size)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.global_id == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllGather: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")

    def profile(self):
        for i in range(30):
            self.profile_allreduce(1 << i, cp.float32, [[0, 1, 2, 3]])
            self.profile_allgather(1 << i, cp.float32, [[0, 1, 2, 3]])
            print("")
            #self.profile_allgather(1 << i, cp.float32, [[0, 1, 2, 3]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 4], [1, 5], [2, 6], [3, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 2, 4, 6], [1, 3, 5, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 1, 2, 3], [4, 5, 6, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 1, 2, 3, 4, 5, 6, 7]])

    def sync(self):
        return


if __name__ == "__main__":
    ray.init(address="auto")

    num_gpus = int(ray.cluster_resources()["GPU"])

    nccl_uuid_list = [cp.cuda.nccl.get_unique_id() for _ in range(500)]

    workers = []
    for i in range(num_gpus):
        workers.append(GpuHost.remote(i, nccl_uuid_list))

    ray.get([w.profile.remote() for w in workers])
    ray.get([w.sync.remote() for w in workers])


