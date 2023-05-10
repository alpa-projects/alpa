"""Utility functions for device mesh workers to call nccl APIs."""
import logging
from typing import Sequence

import cupy
import jax.numpy as jnp
from jax import device_put
from jax._src.dlpack import from_dlpack, to_dlpack
from jax.lib import (
    xla_bridge as xb,
    xla_client as xc,
)
import numpy as np

import alpa.collective as col
from alpa.collective.collective_group import nccl_util
from alpa.util import (jax_tensor_set, jax_tensor_index,
                       xla_buffer_to_jax_tensor, jax_tensor_to_xla_buffer,
                       is_continuous_subset, infer_offset_and_n_elements)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Note: in this device mesh code, we will use 3 types of tensors:
# (1) JAX high-level _DeviceArray, which is index-able, has __cuda_array__
#     interface
# (2) XLA low-level PyLocalBuffer, which is not index-able
# (3) cupy array, which is an intermediate format for ray collective
def send_tile(worker, uuid: int, device_id: int, offset: Sequence[slice],
              dst_rank: int, dst_gpu_idx: int, group_name: str):
    """
    Send a slice of a source buffer to a target GPU.

    Args:
        uuid: the uuid of the xla buffers.
        device_id: the device where the buffer is sent.
        offset: the slice to be sent in the buffer.
        dst_rank: destination rank to send.
        dst_gpu_idx: the gpu index on the destination rank.
        group_name: collective group name
    """
    buffer = worker.buffers[uuid][device_id]
    tensor_shape = buffer.shape
    if is_continuous_subset(offset, tensor_shape):
        # fast path, two cases: (1) same shape, (2) continuous subset.
        slice_shape = tuple(ind.stop - ind.start for ind in offset)
        to_send = xla_buffer_to_cupy(buffer)
        if slice_shape == tensor_shape:
            col.send_multigpu(to_send, dst_rank, dst_gpu_idx, group_name)
        else:
            ind, n_elements = infer_offset_and_n_elements(offset)
            col.send_multigpu(to_send[ind],
                              dst_rank,
                              dst_gpu_idx,
                              group_name,
                              n_elements=n_elements)
    else:
        # slower path, because of indexing.
        logger.debug("Send goes along the slowest path. "
                     "If this is for transformers, please check the resharding "
                     "specs.")
        start_indices = tuple(o.start for o in offset)
        slice_sizes = tuple(o.stop - o.start for o in offset)
        src_buffer = jax_tensor_index(xla_buffer_to_jax_tensor(buffer),
                                      start_indices, slice_sizes)
        to_send = jax_tensor_to_cupy(src_buffer)
        col.send_multigpu(to_send, dst_rank, dst_gpu_idx, group_name)


def recv_tile(worker, uuid: int, device_id: int,
              indices_in_dst_tile: Sequence[slice], src_rank: int,
              src_gpu_idx: int, group_name: str):
    """
    Receive a slice from a source GPU and in-place write it on the target
    buffer.

    Args:
        uuid: the uuid of the xla buffers.
        device_id: the device where the buffer is received, used to allocate
            tmp buffer.
        indices_in_dst_tile: the slice index to be written on destination
            buffer.
        src_rank: source rank to receive from.
        src_gpu_idx: the sender gpu index on the source rank.
        group_name: collective group name.
    """

    buffer = worker.buffers[uuid][device_id]
    tensor_shape = buffer.shape
    slice_shape = tuple(ind.stop - ind.start for ind in indices_in_dst_tile)
    is_bool = buffer.dtype == np.bool_
    if is_continuous_subset(indices_in_dst_tile, tensor_shape):
        to_recv = xla_buffer_to_cupy(buffer, take_ownership=True)
        if slice_shape == tensor_shape:
            col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
        else:
            ind, n_elements = infer_offset_and_n_elements(indices_in_dst_tile)
            col.recv_multigpu(to_recv[ind],
                              src_rank,
                              src_gpu_idx,
                              group_name,
                              n_elements=n_elements)
        new_buffer = cupy_to_xla_buffer(to_recv)
    else:
        # The following call will allocate memory and cause a few H2D and
        # D2D kernels.
        # See: https://github.com/alpa-projects/alpa/issues/145
        logger.debug("Recv goes along the slowest path. "
                     "If this is for transformers, please check the resharding "
                     "specs.")
        tmp_buffer = device_put(jnp.ones(slice_shape, dtype=buffer.dtype),
                                worker.local_devices[device_id])
        to_recv = jax_tensor_to_cupy(tmp_buffer, take_ownership=True)
        col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
        recv_tensor = cupy_to_jax_tensor(to_recv)
        start_indices = tuple(
            ind_in_dst.start for ind_in_dst in indices_in_dst_tile)

        # The following in-place write will cause a D2D copy kernel
        # See: https://github.com/alpa-projects/alpa/issues/144
        # It is unavoidable, but it is better than:
        # new_buffer = dynamic_update_slice(src_buf, update, start_indices)
        # which is not in-place and will cause extra allocation-related
        # kernels.
        new_buffer = jax_tensor_set(xla_buffer_to_jax_tensor(buffer),
                                    recv_tensor, start_indices)
        new_buffer = jax_tensor_to_xla_buffer(new_buffer)
    if is_bool:
        new_buffer = _uint8_to_bool(new_buffer)
    worker.buffers[uuid][device_id] = new_buffer


def allgather(worker, uuid: int, device_ids: Sequence[int],
              tensor_slices: Sequence[Sequence[slice]], output_slice):
    cupy_buffers = []
    communicators = worker.allgather_communicators[repr(sorted(device_ids))]
    relative_idx = dict(zip(sorted(device_ids), range(len(device_ids))))
    output_idx, _ = infer_offset_and_n_elements(output_slice)
    is_bool = worker.buffers[uuid][0].dtype == np.bool_
    nccl_util.groupStart()
    for device_id, tensor_slice in zip(device_ids, tensor_slices):
        xla_buffer = worker.buffers[uuid][device_id]
        cupy_buffer = xla_buffer_to_cupy(xla_buffer, take_ownership=True)
        ind, n_elements = infer_offset_and_n_elements(tensor_slice)
        cupy_slice = cupy_buffer[ind]
        cupy_output_slice = cupy_buffer[output_idx]
        communicators[relative_idx[device_id]].allGather(
            nccl_util.get_tensor_ptr(cupy_slice),
            nccl_util.get_tensor_ptr(cupy_output_slice), n_elements,
            nccl_util.get_nccl_tensor_dtype(cupy_buffer),
            cupy.cuda.Stream.null.ptr)
        cupy_buffers.append(cupy_buffer)
    nccl_util.groupEnd()
    for device_id, cupy_buffer in zip(device_ids, cupy_buffers):
        buf = cupy_to_xla_buffer(cupy_buffer)
        if is_bool:
            buf = _uint8_to_bool(buf)
        worker.buffers[uuid][device_id] = buf


def broadcast(worker, uuid, comm_key, world_size, devices_ids,
              devices_global_rank, tensor_slices, group_name):
    to_use = []
    for_buffer = []
    is_bool = worker.buffers[uuid][devices_ids[0]].dtype == np.bool_
    for device_id, global_rank, tensor_slice in zip(devices_ids,
                                                    devices_global_rank,
                                                    tensor_slices):
        buffer = worker.buffers[uuid][device_id]
        tensor_shape = buffer.shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            # fast path, two cases: (1) same shape, (2) continuous subset.
            tmp = xla_buffer_to_cupy(buffer)
            if slice_shape != tensor_shape:
                ind, _ = infer_offset_and_n_elements(tensor_slice)
                to_use.append(tmp[ind])
            else:
                to_use.append(tmp)
            for_buffer.append(tmp)
        else:
            tmp = None
            if global_rank == 0:
                start_indices = tuple(o.start for o in tensor_slice)
                tmp = jax_tensor_index(xla_buffer_to_jax_tensor(buffer),
                                       start_indices, slice_shape)
                tmp = jax_tensor_to_cupy(tmp)
            else:
                tmp = device_put(jnp.ones(slice_shape, dtype=buffer.dtype),
                                 worker.local_devices[device_id])
                tmp = jax_tensor_to_cupy(tmp, take_ownership=True)
            to_use.append(tmp)
            for_buffer.append(tmp)

    _, n_elements = infer_offset_and_n_elements(tensor_slices[0])
    col.broadcast_partialgpu(to_use, n_elements, comm_key, world_size,
                             devices_ids, devices_global_rank, group_name)

    for for_buffer_tensor, device_id, global_rank, tensor_slice in zip(
            for_buffer, devices_ids, devices_global_rank, tensor_slices):
        if global_rank == 0:
            continue
        buffer = worker.buffers[uuid][device_id]
        tensor_shape = buffer.shape
        slice_shape = tuple(ind.stop - ind.start for ind in tensor_slice)
        if is_continuous_subset(tensor_slice, tensor_shape):
            new_buffer = cupy_to_xla_buffer(for_buffer_tensor)
        else:
            recv_tensor = cupy_to_jax_tensor(for_buffer_tensor)
            start_indices = tuple(
                ind_in_dst.start for ind_in_dst in tensor_slice)
            new_buffer = jax_tensor_set(xla_buffer_to_jax_tensor(buffer),
                                        recv_tensor, start_indices)
            new_buffer = jax_tensor_to_xla_buffer(new_buffer)
        if is_bool:
            new_buffer = _uint8_to_bool(new_buffer)
        worker.buffers[uuid][device_id] = new_buffer


def to_signal_buffer(jax_tensor):
    return jax_tensor_to_cupy(jax_tensor, take_ownership=True)


def xla_buffer_to_cupy(xla_buf, take_ownership=False):
    """Convert an xla buffer directly to cupy, w/o transitioning from jax
    buffer."""
    return cupy.fromDlpack(
        xc._xla.buffer_to_dlpack_managed_tensor(  # pylint: disable=protected-access
            xla_buf,
            take_ownership=take_ownership))


def cupy_to_xla_buffer(tensor):
    """Convert cupy tensors to XLA buffers."""
    if isinstance(tensor, list):
        return list(map(cupy_to_xla_buffer, tensor))
    cpu_backend = xb.get_backend("cpu")
    try:
        gpu_backend = xb.get_backend("gpu")
    except RuntimeError:
        gpu_backend = None
    buf = xc._xla.dlpack_managed_tensor_to_buffer(  # pylint: disable=protected-access
        tensor.toDlpack(), cpu_backend, gpu_backend)
    return buf


def jax_tensor_to_cupy(tensors, take_ownership=False):
    """Convert a Jax DeviceArray to cupy tensor; zero copy."""
    if isinstance(tensors, list):
        return list(map(jax_tensor_to_cupy, tensors))
    return cupy.fromDlpack(to_dlpack(tensors, take_ownership=take_ownership))


def cupy_to_jax_tensor(tensors):
    """Convert cupy tensors to JAX tensors."""
    if isinstance(tensors, list):
        return list(map(cupy_to_jax_tensor, tensors))
    return from_dlpack(tensors.toDlpack())


# in XLA pred(bool) and uint8 are different, but xla->dlpack->xla
# turns a bool into uint8. This implementation is slow.
def _uint8_to_bool(xla_buffer):
    buf = xla_buffer_to_jax_tensor(xla_buffer).astype(np.bool_)
    return jax_tensor_to_xla_buffer(buf)
