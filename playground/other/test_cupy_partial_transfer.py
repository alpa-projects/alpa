import time

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray


# tensor = cp.random.normal(size=[2, 1025, 1536])
# print(tensor)
#
# row_major = True
# print(tensor.data.ptr + 2)
# print(tensor.data.ptr + 2)

MB = 1 << 20
GB = 1 << 30


def do_send_recv(comm, buf, is_sender):
    if is_sender:
        comm.send(buf[2,:].data.ptr, buf.size / 2, nccl.NCCL_FLOAT32,
                  1, cp.cuda.Stream.null.ptr)
    else:
        comm.recv(buf[2,:].data.ptr, buf.size / 2, nccl.NCCL_FLOAT32,
                  0, cp.cuda.Stream.null.ptr)


@ray.remote(num_gpus=1)
class GpuHost:
    def __init__(self, rank, world_size, nccl_uuid_list):
        self.rank = rank
        self.world_size = world_size
        self.nccl_uuid_list = nccl_uuid_list
        self.ct = 0

    def init_communicator(self, groups):
        comm = None
        for group in groups:
            nccl_uuid = self.nccl_uuid_list[self.ct]
            self.ct += 1
            for device_id in group:
                if self.rank == device_id:
                    assert comm is None
                    comm = cp.cuda.nccl.NcclCommunicator(
                        len(group), nccl_uuid, group.index(self.rank))


        cp.cuda.Device(0).synchronize()
        return comm

    def profile_send_recv(self, size, dtype, from_rank, to_rank):
        groups = [[from_rank, to_rank]]
        comm = self.init_communicator(groups)
        if comm is None:
            return

        if self.rank == from_rank:
            buf = cp.zeros((size, size), dtype)
        else:
            buf = cp.ones((size, size), dtype)

        if self.rank == to_rank:
            print("Before send/recv: ", buf)
        do_send_recv(comm, buf, self.rank == from_rank)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank == from_rank)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == from_rank:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")
        if self.rank == to_rank:
            print("After send/recv: ", buf)

    def profile(self):
        # All-reduce

        # Send-recv
        # for i in range(5, 6):
        self.profile_send_recv(1 << 3, cp.float32, 0, 1)
        self.profile_send_recv(1 << 3, cp.float32, 0, self.world_size - 1)


    def sync(self):
        return


if __name__ == "__main__":
    ray.init(address="auto")

    num_gpus = int(ray.cluster_resources()["GPU"])

    nccl_uuid_list = [cp.cuda.nccl.get_unique_id() for _ in range(500)]

    workers = []
    for i in range(num_gpus):
        env_vars = {
            #"NCCL_SOCKET_NTHREADS": "4",
            #"NCCL_NSOCKS_PERTHREAD": "8",
            #"NCCL_ALGO": "tree",
            #"NCCL_DEBUG": "INFO",
        }
        workers.append(GpuHost.options(runtime_env={"env_vars": env_vars}) \
                       .remote(i, num_gpus, nccl_uuid_list))

    ray.get([w.profile.remote() for w in workers])
    ray.get([w.sync.remote() for w in workers])

