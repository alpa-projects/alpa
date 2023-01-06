"""
Benchmark the communication bandwidth with Ray + NCCL.
We use the python binding cupy.nccl to call NCCL.

Usage:
  python3 profile_communication.py
"""

import argparse
import time
import os

import cupy as cp
from cupy.cuda import nccl
import numpy as np
import ray

MB = 1 << 20
GB = 1 << 30


def do_all_reduce(comm, in_buffer, out_buffer):
    comm.allReduce(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        0,
        cp.cuda.Stream.null.ptr,
    )


def do_all_gather(comm, in_buffer, out_buffer):
    comm.allGather(
        in_buffer.data.ptr,
        out_buffer.data.ptr,
        in_buffer.size,
        nccl.NCCL_FLOAT32,
        cp.cuda.Stream.null.ptr,
    )


def do_send_recv(comm, buf, is_sender):
    if is_sender:
        comm.send(buf.data.ptr, buf.size, nccl.NCCL_FLOAT32,
                  1, cp.cuda.Stream.null.ptr)
    else:
        comm.recv(buf.data.ptr, buf.size, nccl.NCCL_FLOAT32,
                  0, cp.cuda.Stream.null.ptr)


@ray.remote(num_gpus=1)
class GpuHost:
    def __init__(self, rank, world_size, nccl_uuid_list):
        self.rank = rank
        self.world_size = world_size
        self.nccl_uuid_list = nccl_uuid_list
        self.ct = 0

    def init_communicator(self, groups):
        if np.max(groups) >= self.world_size:
            return None
        if len(set(np.ravel(groups))) < len(np.ravel(groups)):
            return None

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

    def profile_allreduce(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_reduce(comm, in_buffer, out_buffer)
        do_all_reduce(comm, in_buffer, out_buffer)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_reduce(comm, in_buffer, out_buffer)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = 2 * array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllReduce: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")

    def profile_allgather(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        if comm is None:
            return

        in_buffer = cp.ones(int(size) // len(groups[0]), dtype)
        out_buffer = cp.ones(int(size), dtype)

        do_all_gather(comm, in_buffer, out_buffer)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_all_gather(comm, in_buffer, out_buffer)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == 0:
            num_devices = len(groups[0])
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size * (num_devices - 1) / num_devices
            bandwidth = communication_size / time_cost
            print(f"AllGather: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")

    def profile_send_recv(self, size, dtype, from_rank, to_rank):
        groups = [[from_rank, to_rank]]
        comm = self.init_communicator(groups)
        if comm is None:
            return

        buf = cp.ones(int(size), dtype)
        do_send_recv(comm, buf, self.rank == from_rank)
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

    def profile_multi_send_recv(self, size, dtype, groups):
        comm = self.init_communicator(groups)
        time.sleep(1)
        comm_sync = self.init_communicator([list(np.ravel(groups))])
        if comm is None or comm_sync is None:
            return

        assert all(len(group) == 2 for group in groups)

        senders = set(group[0] for group in groups)
        receivers = set(group[1] for group in groups)

        buf = cp.ones(int(size), dtype)
        buf_sync = cp.ones(1, dtype)

        do_send_recv(comm, buf, self.rank in senders)
        do_send_recv(comm, buf, self.rank in senders)
        do_all_reduce(comm_sync, buf_sync, buf_sync)

        number = min(max(10, int((1 << 30) / (size * dtype().nbytes))), 1 << 13)
        cp.cuda.Device(0).synchronize()
        tic = time.time()
        for i in range(number):
            do_send_recv(comm, buf, self.rank in senders)
        do_all_reduce(comm_sync, buf_sync, buf_sync)
        cp.cuda.Device(0).synchronize()
        toc = time.time()

        if self.rank == groups[0][0]:
            time_cost = (toc - tic) / number
            array_size = size * dtype().nbytes
            communication_size = array_size
            bandwidth = len(groups) * communication_size / time_cost
            print(f"SendRecv: {groups}\tBytes: {array_size / GB:.5f} GB\t"
                  f"Time: {time_cost:.5f} s\tBandwidth: {bandwidth / (1<<30):.2f} GB/s")

    def profile(self):
        # All-reduce
        for i in range(29, 30):
            self.profile_allreduce(1 << i, cp.float32, [list(range(self.world_size))])
            self.profile_allreduce(1 << i, cp.float32, [list(range(self.world_size//2))])

            #self.profile_allreduce(1 << i, cp.float32, [[0, 3]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 4], [1, 5], [2, 6], [3, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 2, 4, 6], [1, 3, 5, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 1, 2, 3], [4, 5, 6, 7]])
            #self.profile_allreduce(1 << i, cp.float32, [[0, 1, 2, 3, 4, 5, 6, 7]])

        # single Send-recv
        for i in range(29, 30):
            self.profile_send_recv(1 << i, cp.float32, 0, 1)
            self.profile_send_recv(1 << i, cp.float32, 0, self.world_size - 1)

        # multiple p2p Send-recv
        for i in range(29, 30):
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, 1], [2, 3]])
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3]])
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 2], [1, self.world_size - 1]])
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 4], [1, self.world_size - 3], [2, self.world_size - 2], [3, self.world_size - 1]])
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5]])
            self.profile_multi_send_recv(1 << i, cp.float32, [[0, self.world_size - 8], [1, self.world_size - 7], [2, self.world_size - 6], [3, self.world_size - 5],
                                                              [4, self.world_size - 4], [5, self.world_size - 3], [6, self.world_size - 2], [7, self.world_size - 1]])

    def sync(self):
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--efa", action="store_true",
        help="Use AWS EFS on p3.24 or p4.24 instances")
    parser.add_argument("--ib", action="store_true",
        help="Use InfiniBand for NCCL communcation")
    parser.add_argument("--debug", action="store_true",
        help="Print nccl debug information")
    args = parser.parse_args()

    ray.init(address="auto")
    num_gpus = int(ray.cluster_resources()["GPU"])

    nccl_uuid_list = [cp.cuda.nccl.get_unique_id() for _ in range(500)]

    workers = []
    for i in range(num_gpus):
        if args.efa:
            env_vars = {
                "FI_PROVIDER": "efa",
                "FI_EFA_USE_DEVICE_RDMA": "1",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),  # For libnccl-net.so
            }
        elif args.ib:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "NCCL_IB_HCA": "ibp",  # Change this to align with your IB interface name
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }
        else:
            env_vars = {
                "NCCL_SOCKET_NTHREADS": "4",
                "NCCL_NSOCKS_PERTHREAD": "4",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }

        if args.debug:
            env_vars["NCCL_DEBUG"] = "INFO"

        workers.append(GpuHost.options(runtime_env={"env_vars": env_vars})\
                              .remote(i, num_gpus, nccl_uuid_list))

    ray.get([w.profile.remote() for w in workers])
    ray.get([w.sync.remote() for w in workers])
