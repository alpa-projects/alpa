import time

import numpy as np

import cupy as cp
from cupy.cuda import nccl

MB = 1 << 20
GB = 1 << 30

def allocate_buffers(gpu_ids, size, dtype):
    buffers = []
    for gpu_id in gpu_ids:
        cp.cuda.Device(gpu_id).use()
        buffers.append(cp.ones(size, dtype))
    return buffers


def sync(gpu_ids):
    for gpu_id in gpu_ids:
        cp.cuda.Device(gpu_id).synchronize()


def do_all_reduce(comms, in_buffers, out_buffers, size, gpu_ids):
    for ct, gpu_id in enumerate(gpu_ids):
        cp.cuda.Device(gpu_id).use()
        comms[ct].allReduce(
            in_buffers[ct].data.ptr,
            out_buffers[ct].data.ptr,
            size,
            nccl.NCCL_FLOAT32,
            0,
            cp.cuda.Stream.null.ptr,
        )


def benchmark(size, dtype, groups):
    comms = []
    for group in groups:
        comms.extend(nccl.NcclCommunicator.initAll(group))

    gpu_ids = np.array(groups).flatten()

    in_buffers = allocate_buffers(gpu_ids, size, dtype)
    out_buffers = allocate_buffers(gpu_ids, size, dtype)

    do_all_reduce(comms, in_buffers, out_buffers, size, gpu_ids)
    do_all_reduce(comms, in_buffers, out_buffers, size, gpu_ids)

    number = 30
    sync(gpu_ids)
    tic = time.time()
    for i in range(number):
        do_all_reduce(comms, in_buffers, out_buffers, size, gpu_ids)
    sync(gpu_ids)
    toc = time.time()

    num_devices = len(groups[0])
    time_cost = (toc - tic) / number
    total_bytes = 2 * size * dtype().nbytes * (num_devices - 1) / (num_devices)
    bandwidth = total_bytes / time_cost
    print(f"Group: {groups}\tBytes: {total_bytes / GB:.3f} GB\t"
          f"Bandwidth: {bandwidth / (1<<30):.2f} GB/s")

benchmark(128 << 20, cp.float32, [[0, 1]])
benchmark(128 << 20, cp.float32, [[2, 3]])
benchmark(128 << 20, cp.float32, [[0, 1], [2, 3]])
benchmark(128 << 20, cp.float32, [[0, 1, 2, 3]])

for size in range(18):
    benchmark(1 << (10 + size), cp.float32, [[0, 1, 2, 3]])

