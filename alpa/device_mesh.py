"""The device mesh runtime that manages buffers and runs computation distributedly."""
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from collections.abc import Iterable
from functools import partial
from itertools import chain
import logging
from operator import attrgetter
import os
import time
from typing import Any, List, Union, Sequence, Tuple, Optional

import cupy
from cupy.cuda import nccl
import jax
from jax import core, jit, xla, device_put
from jax._src.api import ShapeDtypeStruct
from jax._src.lib import xla_bridge as xb, xla_extension as xe
from jax._src.numpy.lax_numpy import _multi_slice
from jax._src.tree_util import tree_leaves
from jax._src.util import unzip3
from jax.abstract_arrays import array_types
from jax.core import ShapedArray
from jax.interpreters import pxla
from jax.interpreters.pxla import (ShardingSpec, _as_slice_indices,
                                   _hashable_index, ShardedDeviceArray, Index)
from jax.lib import xla_client
import jax.numpy as jnp
import numpy as np
import ray

from alpa import mesh_profiling
import alpa.collective as col
from alpa.collective.collective_group import nccl_util
from alpa.global_env import global_config
from alpa.monkey_patch import set_override_backend
from alpa.shard_parallel.auto_sharding import LogicalDeviceMesh
from alpa.timer import timers
from alpa.util import (benchmark_func, get_microbatch_sharding_spec,
                       list_gpu_info, jax_tensor_to_cupy, cupy_to_jax_tensor,
                       jax_tensor_set, xla_buffer_to_jax_tensor,
                       jax_tensor_to_xla_buffer, xla_buffer_to_cupy,
                       cupy_to_xla_buffer, is_continuous_subset,
                       infer_offset_and_n_elements, jax_tensor_index)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ReshardingTileSpec = namedtuple("ReshardingSendSpec",
                                ["offset", "rank", "gpu_idx"])
ReshardingSendTask = namedtuple("ReshardingSendTask",
                                ["tile_specs", "group_name"])
ReshardingRecvSpec = namedtuple("ReshardingRecvSpec",
                                ["device_id", "shape", "dtype", "tile_specs"])
ReshardingRecvTask = namedtuple("ReshardingRecvTask",
                                ["recv_specs", "group_name"])
ReshardingAllGatherSpec = namedtuple("ReshardingAllGatherSpec",
                                     ["device_ids", "tensor_slices"])
ReshardingAllGatherTask = namedtuple("ReshardingAllGatherTask",
                                     ["allgather_specs"])


class MeshHostWorker:
    """A ray actor that manages the xla computation and buffers on a single host."""

    def __init__(self, server_address: str, num_hosts: int, host_id: int):
        self.num_hosts = num_hosts
        self.host_id = host_id
        self.distributed_client = (
            xla_client._xla.get_distributed_runtime_client(
                server_address, host_id))
        logger.debug(
            f"{host_id}: Trying to connect to xla runtime at {server_address}")
        status = self.distributed_client.connect()
        logger.debug(
            f"{host_id}: Success to connect to xla runtime at {server_address}")
        self.backend = xla_client.make_gpu_client(self.distributed_client,
                                                  node_id=host_id)
        # Monkey patch the backend
        self.local_devices = self.backend.local_devices()
        self.allgather_communicators = {}
        self.buffers = {}  # Dict[uuid -> DeviceArray]
        self.executables = {}  # Dict[uud -> MeshWorkerExecutable]
        self.send_tasks = {}  # Dict[uuid -> ReshardingSendTask]
        self.recv_tasks = {}  # Dict[uuid -> ReshardingRecvTask]
        self.allgather_tasks = {}  # Dict[uuid -> AllgatherTask]
        self.data_loaders = {}  # Dict[uuid -> MeshWorkerDataLoader]
        self.data_loader_iters = {}  # Dict[uuid -> iterator]
        set_override_backend(self.backend)

        if global_config.pipeline_use_signal_send_recv:
            print("Use signal send recv.")
            self.signal_tensors = []
            for d in self.local_devices:
                self.signal_tensors.append(
                    jax_tensor_to_cupy(device_put(
                        jnp.ones((1,), dtype=jnp.int8), d),
                                       take_ownership=True))

    ##### Buffer Related Functions #####
    def put_buffer(self, uuid: int, device_id: int, data: np.ndarray):
        assert uuid not in self.buffers
        if data.dtype == np.int64:
            data = data.astype(np.int32)
        self.buffers[uuid] = (self.backend.buffer_from_pyval(
            data, self.local_devices[device_id]))

    def put_buffers(self,
                    uuids: Sequence[int],
                    device_ids: Sequence[int],
                    datas: Sequence[np.ndarray],
                    num_batch=1,
                    batch_dim=0):
        if num_batch > 1:
            device_ids = list(chain(*[[x] * num_batch for x in device_ids]))
            split_datas = []
            for data in datas:
                split_buffers = np.split(data, num_batch, batch_dim)
                split_datas.extend(split_buffers)
            datas = split_datas
        for uuid, device_id, data in zip(uuids, device_ids, datas):
            if data.dtype == np.int64:
                data = data.astype(np.int32)
            self.buffers[uuid] = (self.backend.buffer_from_pyval(
                data, self.local_devices[device_id]))

    def put_non_zero_buffer(self,
                            uuid: int,
                            device_id: int,
                            shape: Sequence[int],
                            dtype=np.float32):
        if dtype == np.int64:
            dtype = np.int32
        self.buffers[uuid] = (self.backend.buffer_from_pyval(
            np.full(shape, 1e-8, dtype), self.local_devices[device_id]))

    def shard_and_put_non_zero_buffer(self, uuids, shape, dtype, indices,
                                      num_batch):
        assert len(uuids) == len(indices) == len(self.local_devices) * num_batch
        for i in range(len(self.local_devices)):
            for b in range(num_batch):
                shard_shape = []
                idx = i * num_batch + b
                for j, s in enumerate(indices[idx]):
                    filled_slice = s.indices(shape[j])
                    dim_size = len(range(*filled_slice))
                    shard_shape.append(dim_size)
                self.put_non_zero_buffer(uuids[idx], i, shard_shape, dtype)

    def get_buffers(self, uuids: Union[Sequence[int], int]):
        if isinstance(uuids, Iterable):
            return [self.buffers[uuid] for uuid in uuids]
        return self.buffers[uuids]

    def delete_buffers(self, uuids: Union[Sequence[int], int]):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                del self.buffers[uuid]
        else:
            del self.buffers[uuids]

    def block_until_ready_buffers(self, uuids: Union[Sequence[int], int]):
        if isinstance(uuids, Iterable):
            for uuid in uuids:
                self.buffers[uuid].block_until_ready()
        else:
            self.buffers[uuids].block_until_ready()

    def get_memory_allocated(self):
        self.sync()
        return max(d.memory_allocated() for d in self.local_devices)

    def get_max_memory_allocated(self):
        self.sync()
        return max(d.max_memory_allocated() for d in self.local_devices)

    def get_available_memory(self):
        self.sync()
        return min(d.available_memory() for d in self.local_devices)

    def reset_memory_stats(self):
        self.sync()
        for device in self.local_devices:
            device.clear_memory_stats()

    ##### Executable Related Functions #####
    def put_executable(self, uuid: int, executable_class, *args):
        self.executables[uuid] = executable_class(self, uuid, *args)

    def delete_executable(self, uuid: int):
        del self.executables[uuid]

    def run_executable(self, uuid: int, *args, **kwargs):
        self.executables[uuid].execute_on_worker(*args, **kwargs)

    def get_exec_hlo_text(self, uuid: int):
        return self.executables[uuid].get_hlo_text()

    def get_exec_total_allocation_size(self, uuid: int):
        return self.executables[uuid].get_total_allocation_size()

    def get_exec_grad_sync_channel_ids(self, uuid: int):
        return self.executables[uuid].grad_sync_channel_ids

    ##### Data loader Related Functions #####
    def put_data_loader(self, uuid: int, *args):
        from alpa.data_loader import MeshWorkerDataLoader
        self.data_loaders[uuid] = MeshWorkerDataLoader(self, *args)

    def data_loader_iter(self, uuid: int):
        self.data_loader_iters[uuid] = iter(self.data_loaders[uuid])

    def data_loader_next(self, uuid: int):
        next(self.data_loader_iters[uuid])

    ##### Cross Mesh Resharding Related Functions #####
    @staticmethod
    def init_collective_group(world_size, rank, backend, group_name):
        """Initialize the collective group eagerly."""
        col.init_collective_group(world_size,
                                  rank,
                                  backend=backend,
                                  group_name=group_name)

    # Note: in this device mesh code, we will use 3 types of tensors:
    # (1) JAX high-level _DeviceArray, which is index-able, has __cuda_array__ interface
    # (2) XLA low-level PyLocalBuffer, which is not index-able
    # (3) cupy array, which is an intermediate format for ray collective
    def send_tile(self, uuid: int, offset: Sequence[slice], dst_rank: int,
                  dst_gpu_idx: int, group_name: str):
        """
        Send a slice of a source buffer to a target GPU.

        Args:
            uuid: the uuid of the xla buffers.
            offset: the slice to be sent in the buffer.
            dst_rank: destination rank to send.
            dst_gpu_idx: the gpu index on the destination rank.
            group_name: collective group name
        """
        if global_config.pipeline_use_signal_send_recv:
            signal = self.signal_tensors[uuid % len(self.local_devices)]
            col.send_multigpu(signal, dst_rank, dst_gpu_idx, group_name)
            return

        tensor_shape = self.buffers[uuid].shape
        if is_continuous_subset(offset, tensor_shape):
            # fast path, two cases: (1) same shape, (2) continuous subset.
            slice_shape = tuple(ind.stop - ind.start for ind in offset)
            to_send = xla_buffer_to_cupy(self.buffers[uuid])
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
            logger.debug(
                "Send goes along the slowest path. "
                "If this is for transformers, please check the resharding specs."
            )
            start_indices = tuple(o.start for o in offset)
            slice_sizes = tuple(o.stop - o.start for o in offset)
            src_buffer = jax_tensor_index(
                xla_buffer_to_jax_tensor(self.buffers[uuid]), start_indices,
                slice_sizes)
            to_send = jax_tensor_to_cupy(src_buffer)
            col.send_multigpu(to_send, dst_rank, dst_gpu_idx, group_name)

    def recv_tile(self, uuid: int, device_id: int,
                  indices_in_dst_tile: Sequence[slice], src_rank: int,
                  src_gpu_idx: int, group_name: str):
        """
        Receive a slice from a source GPU and in-place write it on the target buffer.

        Args:
            uuid: the uuid of the xla buffers.
            device_id: the device where the buffer is received, used to allocate tmp buffer.
            indices_in_dst_tile: the slice index to be written on destination buffer.
            src_rank: source rank to receive from.
            src_gpu_idx: the sender gpu index on the source rank.
            group_name: collective group name.
        """
        if uuid not in self.buffers:
            raise RuntimeError("Buffer has not been created.")

        if global_config.pipeline_use_signal_send_recv:
            signal = self.signal_tensors[uuid % len(self.local_devices)]
            col.recv_multigpu(signal, src_rank, src_gpu_idx, group_name)
            return

        tensor_shape = self.buffers[uuid].shape
        slice_shape = tuple(ind.stop - ind.start for ind in indices_in_dst_tile)
        is_bool = self.buffers[uuid].dtype == np.bool_
        if is_continuous_subset(indices_in_dst_tile, tensor_shape):
            to_recv = xla_buffer_to_cupy(self.buffers[uuid],
                                         take_ownership=True)
            if slice_shape == tensor_shape:
                col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
            else:
                ind, n_elements = infer_offset_and_n_elements(
                    indices_in_dst_tile)
                col.recv_multigpu(to_recv[ind],
                                  src_rank,
                                  src_gpu_idx,
                                  group_name,
                                  n_elements=n_elements)
            self.buffers[uuid] = cupy_to_xla_buffer(to_recv)
        else:
            # The following call will allocate memory and cause a few H2D and D2D kernels.
            # See:https://github.com/alpa-projects/alpa/issues/145
            logger.debug(
                "Recv goes along the slowest path. "
                "If this is for transformers, please check the resharding specs."
            )
            tmp_buffer = device_put(
                jnp.ones(slice_shape, dtype=self.buffers[uuid].dtype),
                self.local_devices[device_id])
            to_recv = jax_tensor_to_cupy(tmp_buffer, take_ownership=True)
            col.recv_multigpu(to_recv, src_rank, src_gpu_idx, group_name)
            recv_tensor = cupy_to_jax_tensor(to_recv)
            start_indices = tuple(
                ind_in_dst.start for ind_in_dst in indices_in_dst_tile)

            # The following in-place write will cause a D2D copy kernel
            # See: https://github.com/alpa-projects/alpa/issues/144
            # It is unavoidable, but it is better than:
            # new_buffer = dynamic_update_slice(src_buf, update, start_indices)
            # which is not in-place and will cause extra allocation-related kernels.
            new_buffer = jax_tensor_set(
                xla_buffer_to_jax_tensor(self.buffers[uuid]), recv_tensor,
                start_indices)
            self.buffers[uuid] = jax_tensor_to_xla_buffer(new_buffer)
        if is_bool:
            self.buffers[uuid] = _uint8_to_bool(self.buffers[uuid])

    def init_p2p_communicator(self, group_name, my_rank, my_gpu_idx, peer_rank,
                              peer_gpu_idx, nccl_uid):
        """Initialize the P2P communicator from within the mesh workers."""
        assert col.is_group_initialized(group_name)
        assert col.get_rank(group_name) == my_rank
        g = col.check_and_get_group(group_name)
        g.create_p2p_communicator(my_gpu_idx, peer_rank, peer_gpu_idx, nccl_uid)

    def generate_nccl_uid(self, group_name):
        """Generate the NCCL unique ID in advance."""
        g = col.check_and_get_group(group_name)
        uid = g.generate_nccl_uid()
        return uid

    def allgather(self, uuids: Sequence[int], device_ids: Sequence[int],
                  tensor_slices: Sequence[slice]):
        cupy_buffers = []
        communicators = self.allgather_communicators[repr(sorted(device_ids))]
        relative_idx = dict(zip(sorted(device_ids), range(len(device_ids))))
        is_bool = self.buffers[uuids[0]].dtype == np.bool_
        nccl_util.groupStart()
        for device_id, tensor_slice in zip(device_ids, tensor_slices):
            uuid = uuids[device_id]
            xla_buffer = self.buffers[uuid]
            cupy_buffer = xla_buffer_to_cupy(xla_buffer, take_ownership=True)
            ind, n_elements = infer_offset_and_n_elements(tensor_slice)
            cupy_slice = cupy_buffer[ind]
            communicators[relative_idx[device_id]].allGather(
                nccl_util.get_tensor_ptr(cupy_slice),
                nccl_util.get_tensor_ptr(cupy_buffer), n_elements,
                nccl_util.get_nccl_tensor_dtype(cupy_buffer),
                cupy.cuda.Stream.null.ptr)
            cupy_buffers.append(cupy_buffer)
        nccl_util.groupEnd()
        for device_id, cupy_buffer in zip(device_ids, cupy_buffers):
            uuid = uuids[device_id]
            buf = cupy_to_xla_buffer(cupy_buffer)
            if is_bool:
                buf = _uint8_to_bool(buf)
            self.buffers[uuid] = buf

    def put_resharding_send_task(self, uuid, tasks, group_name):
        self.send_tasks[uuid] = ReshardingSendTask(tile_specs=tasks,
                                                   group_name=group_name)

    def put_resharding_recv_task(self, uuid, tasks, group_name):
        self.recv_tasks[uuid] = ReshardingRecvTask(recv_specs=tasks,
                                                   group_name=group_name)

    def run_resharding_send_task(self, uuid, buf_uuids):
        task: ReshardingSendTask = self.send_tasks[uuid]
        for send_tile_spec, buf_uuid in zip(task.tile_specs, buf_uuids):
            send_tile_spec: ReshardingTileSpec
            self.send_tile(buf_uuid, send_tile_spec.offset, send_tile_spec.rank,
                           send_tile_spec.gpu_idx, task.group_name)

    def run_resharding_recv_task(self, uuid, buf_uuids, set_empty_buffer=True):
        task: ReshardingRecvTask = self.recv_tasks[uuid]
        for recv_spec, buf_uuid in zip(task.recv_specs, buf_uuids):
            recv_spec: ReshardingRecvSpec
            if set_empty_buffer:
                self.put_non_zero_buffer(buf_uuid, recv_spec.device_id,
                                         recv_spec.shape, recv_spec.dtype)
            for recv_tile_spec in recv_spec.tile_specs:
                recv_tile_spec: ReshardingTileSpec
                self.recv_tile(buf_uuid, recv_spec.device_id,
                               recv_tile_spec.offset, recv_tile_spec.rank,
                               recv_tile_spec.gpu_idx, task.group_name)

    def put_resharding_allgather_task(self, uuid, tasks):
        all_gather_task = ReshardingAllGatherTask(tasks)
        allgather_specs = all_gather_task.allgather_specs
        for group_idx in allgather_specs:
            allgather_spec: ReshardingAllGatherSpec = allgather_specs[group_idx]
            device_ids = sorted(allgather_spec.device_ids)
            if repr(device_ids) not in self.allgather_communicators:
                communicators = nccl.NcclCommunicator.initAll(list(device_ids))
                self.allgather_communicators[repr(device_ids)] = communicators
        self.allgather_tasks[uuid] = all_gather_task

    def run_allgather_task(self, uuid, buffer_uuids):
        task: ReshardingAllGatherTask = self.allgather_tasks[uuid]
        allgather_specs = task.allgather_specs
        for group_idx in allgather_specs:
            allgather_spec: ReshardingAllGatherSpec = allgather_specs[group_idx]
            self.allgather(buffer_uuids, allgather_spec.device_ids,
                           allgather_spec.tensor_slices)

    def destroy_collective_group(self, group_name: str = "default"):
        col.destroy_collective_group(group_name)

    ##### Data Loader Related Functions #####
    def delete_data_loader(self, uuid):
        del self.data_loaders[uuid]

    ##### Profiling Related Functions #####
    def profile_hlo_ops(self, op_infos: Sequence[Any], cache_filename: str,
                        single_timeout: float):
        num_devices = self.num_hosts * len(self.local_devices)
        return mesh_profiling.profile_hlo_ops(op_infos, self.backend,
                                              self.local_devices, self.host_id,
                                              num_devices, cache_filename,
                                              single_timeout)

    def profile_executable_with_dummy_inputs(self, uuid: int, **kwargs):
        return self.executables[uuid].profile_with_dummy_inputs(
            self.backend, self.local_devices, **kwargs)

    def profile_resharding_send_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        # TODO(yonghao): the sync function should be carefully reconsidered
        run_fn = lambda: self.run_resharding_send_task(uuid, buf_uuids)
        sync_fn = self.sync if sync else None
        costs = benchmark_func(run_fn, sync_fn, warmup, repeat, number)
        return np.mean(costs)

    def profile_resharding_recv_task(self,
                                     uuid,
                                     buf_uuids,
                                     warmup=1,
                                     repeat=3,
                                     number=3,
                                     sync=False):
        set_empty_buffer = True

        def run_fn():
            nonlocal set_empty_buffer
            self.run_resharding_recv_task(uuid, buf_uuids, set_empty_buffer)
            set_empty_buffer = False

        sync_fn = self.sync if sync else None
        costs = benchmark_func(run_fn, sync_fn, warmup, repeat, number)
        return np.mean(costs)

    def get_timer(self, name: str):
        return timers(name)

    def reset_timer(self, name: str):
        timers(name).reset()

    ##### Other Functions #####
    def sync(self):
        for device in self.local_devices:
            device.synchronize_all_activity()

    @staticmethod
    def check_alive():
        return 0

    def shutdown(self):
        self.sync()
        del self.buffers
        del self.executables
        self.distributed_client.shutdown()


class PhysicalDeviceMesh(ABC):
    num_devices_per_host: int

    def get_signature(self) -> str:
        """Return a signature string that contains the mesh shape and GPU model."""
        gpu_type = list_gpu_info()
        gpu_name = gpu_type.split("\n")[0].split(" (UUID:")[0][7:]
        ret = f"{self.num_hosts},{self.num_devices_per_host},{gpu_name}"
        ret = ret.replace(" ", "-")
        return ret

    @property
    def shape(self):
        return self.num_hosts, self.num_devices_per_host

    @property
    def num_devices(self):
        """Return the total number of GPUs on this mesh."""
        return self.num_hosts * self.num_devices_per_host

    @property
    @abstractmethod
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        raise NotImplementedError()

    ##### Logical Mesh Related Functions #####
    def get_logical_mesh(self,
                         mesh_shape,
                         mesh_alpha=None,
                         mesh_beta=None,
                         mesh_topology=None,
                         intra_host_bandwidth=None,
                         inter_host_bandwidth=None):
        """Return a logical mesh and parameters of the alpha-beta communication cost model."""
        id_mesh = np.arange(self.num_devices).reshape(mesh_shape)

        if mesh_topology is None:
            mesh_alpha = mesh_alpha or (1,) * len(mesh_shape)
            mesh_beta = mesh_beta or (1,) * len(mesh_shape)
        elif mesh_topology == "tree":
            assert mesh_alpha is None
            assert mesh_beta is None
            mesh_alpha = [1] * 2
            mesh_beta = [None] * 2
            host_ids = np.tile(
                np.arange(self.num_hosts).reshape(-1, 1),
                self.num_devices_per_host)
            host_ids = host_ids.reshape(mesh_shape)

            # Compute bandwidth of doing communication along dim 0.
            # 1. Compute the number of links between each host pairs.
            #    Assume using ring-based algorithms.
            host_link_ct = defaultdict(int)
            for j in range(mesh_shape[1]):
                for i in range(mesh_shape[0]):
                    left = host_ids[i][j]
                    right = host_ids[(i + 1) % mesh_shape[0]][j]
                    if left != right:
                        if left > right:
                            left, right = right, left
                        host_link_ct[(left, right)] += 1

            j = 0
            # 2. Bandwidth between two hosts = total_bandwidth / number_of_links.
            #    Bandwdith along a communication dimension = min bandwidth of all links.
            bandwidth = intra_host_bandwidth
            for i in range(mesh_shape[0]):
                left = host_ids[i][j]
                right = host_ids[(i + 1) % mesh_shape[0]][j]
                if left != right:
                    if left > right:
                        left, right = right, left
                    bandwidth = min(
                        bandwidth,
                        inter_host_bandwidth / host_link_ct[(left, right)])
            mesh_beta[0] = 1 / bandwidth

            # Compute bandwidth of doing communication along dim 1.
            host_link_ct = defaultdict(int)
            for i in range(mesh_shape[0]):
                for j in range(mesh_shape[1]):
                    left = host_ids[i][j]
                    right = host_ids[i][(j + 1) % mesh_shape[1]]
                    if left != right:
                        if left > right:
                            left, right = right, left
                        host_link_ct[(left, right)] += 1

            i = 0
            bandwidth = intra_host_bandwidth
            for j in range(mesh_shape[1]):
                left = host_ids[i][j]
                right = host_ids[i][(j + 1) % mesh_shape[1]]
                if left != right:
                    if left > right:
                        left, right = right, left
                    bandwidth = min(
                        bandwidth,
                        inter_host_bandwidth / host_link_ct[(left, right)])
            mesh_beta[1] = 1 / bandwidth

        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        """Return the default logical mesh."""
        if self.num_hosts == 1:
            return self.get_logical_mesh(
                (self.num_hosts, self.num_devices_per_host), [1, 1], [1, 1])
        else:
            return self.get_logical_mesh(
                (self.num_hosts, self.num_devices_per_host), [1, 1], [1, 0.01])

    ##### Executable Related Functions #####
    @abstractmethod
    def shard_args_to_bufs(self, shard_indices: Sequence[Sequence[Index]],
                           donated_invars: Sequence[bool], args):
        """Shard high-level arguments as low-level buffers."""
        raise NotImplementedError()

    @abstractmethod
    def shard_args_to_arrays(self, avals: Sequence[ShapedArray],
                             shard_indices: Sequence[Sequence[Index]],
                             sharding_specs: Sequence[ShardingSpec], args):
        """Shard arguments (np.ndarray) as distributed arrays."""
        raise NotImplementedError()

    @abstractmethod
    def get_outputs_handler(self, avals: Sequence[ShapedArray],
                            sharding_specs: Sequence[ShardingSpec]):
        """Get a function that wraps low-level buffers to high-level output arrays."""
        raise NotImplementedError()

    ##### Profiling Related Functions #####
    @abstractmethod
    def get_remote_timer(self, timer_name: str):
        raise NotImplementedError()

    @abstractmethod
    def reset_remote_timer(self, timer_name: str):
        raise NotImplementedError()

    @abstractmethod
    def get_memory_allocated(self):
        raise NotImplementedError()

    @abstractmethod
    def get_max_memory_allocated(self):
        raise NotImplementedError()

    @abstractmethod
    def get_available_memory(self):
        raise NotImplementedError()

    @abstractmethod
    def reset_memory_stats(self):
        raise NotImplementedError()

    ##### Other Functions #####
    @abstractmethod
    def sync_workers(self):
        """Sync all device activities on workers."""
        raise NotImplementedError()

    @abstractmethod
    def shutdown(self, forced=False):
        """Shut down the mesh."""
        raise NotImplementedError()


class LocalPhysicalDeviceMesh(PhysicalDeviceMesh):
    """
    A single-host physical device mesh to run computation distributedly. It uses
    the native XLA runtime.
    """

    def __init__(self, devices: Sequence["Device"] = None):
        self.devices = devices if devices is not None else xb.local_devices()
        self.num_devices_per_host = len(self.devices)
        self.device_strs = []

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return 1

    ##### Executable Related Functions #####
    def shard_args_to_bufs(self, shard_indices: Sequence[Sequence[Index]],
                           donated_invars: Sequence[bool], args):
        ret = []
        for arg, donated, indices in zip(args, donated_invars, shard_indices):
            ret.append(pxla._shard_arg(arg, self.devices, indices))
            if isinstance(arg, xe.DeviceArray) and donated:
                arg.delete()
        return ret

    def shard_args_to_arrays(self, avals: Sequence[ShapedArray],
                             shard_indices: Sequence[Sequence[Index]],
                             sharding_specs: Sequence[ShardingSpec], args):
        arrays = []
        for i in range(len(avals)):
            shards = [
                args[i][shard_indices[i][k]] for k in range(len(self.devices))
            ]
            buffers = [
                jax.device_put(x, d) for x, d in zip(shards, self.devices)
            ]
            arrays.append(
                pxla._ShardedDeviceArray(avals[i], sharding_specs[i], buffers,
                                         shard_indices[i]))
        return arrays

    def get_outputs_handler(self, avals: Sequence[ShapedArray],
                            sharding_specs: Sequence[ShardingSpec]):
        outs_handler = pxla.local_avals_to_results_handler(
            sharding_specs, avals)
        return outs_handler

    ##### Profiling Related Functions #####
    def get_remote_timer(self, timer_name: str):
        return timers(timer_name)

    def reset_remote_timer(self, timer_name: str):
        timers(timer_name).reset()

    def get_memory_allocated(self):
        self.sync_workers()
        return max([d.memory_allocated() for d in self.devices])

    def get_max_memory_allocated(self):
        self.sync_workers()
        return max([d.max_memory_allocated() for d in self.devices])

    def get_available_memory(self):
        return min([device.available_memory() for device in self.devices])

    def reset_memory_stats(self):
        for device in self.devices:
            device.clear_memory_stats()

    ##### Other Functions #####
    def sync_workers(self):
        for device in self.devices:
            device.synchronize_all_activity()

    def shutdown(self, forced=False):
        self.sync_workers()


def device_id_to_str(host_ip, device_id, device_type="gpu"):
    """Convert device id (int) to a canonical device string."""
    return "{}:{}:{}".format(host_ip, device_type, str(device_id))


# Used ports for XLA distributed runtime servers.
used_port_set = set((None,))


class DistributedPhysicalDeviceMesh(PhysicalDeviceMesh):
    """
    A multi-host physical device mesh to run computation distributedly. It uses
    ray actors and the distributed XLA runtime.
    """

    def __init__(self,
                 devices: Union[Sequence[Sequence[int]]] = None,
                 host_ids: Sequence[int] = None,
                 host_info: Sequence[dict] = None,
                 head_ip: str = None,
                 num_devices_per_host: int = None):
        self.host_ids = host_ids
        self.host_info = host_info
        self.head_ip = head_ip
        self.num_devices_per_host = num_devices_per_host
        self.workers = None
        self.launched = False

        if devices is not None:
            if len(devices) != len(host_ids):
                raise RuntimeError(
                    "Please specify the gpu IDs used on each host.")
            if not all(len(ids) == num_devices_per_host for ids in devices):
                raise RuntimeError(
                    "Devices specified for each host does not align "
                    "with `num_devices_per_host`.")
        else:
            devices = [
                list(range(num_devices_per_host))
                for i, _ in enumerate(host_ids)
            ]
        self.devices = devices
        self.device_strs = []
        for i in range(self.num_hosts):
            ip = self.host_info[i]["NodeManagerAddress"]
            self.device_strs.extend(
                [device_id_to_str(ip, j) for j in devices[i]])
        self._launch_xla_servers()

        self.to_delete_remote_buffers = [[] for _ in range(self.num_hosts)]
        self.to_delete_remote_buffers_ct = 0

    def _launch_xla_servers(self):
        # Launch distributed xla runtime
        port = None
        while port in used_port_set:
            port = np.random.randint(20000, 25000)
        used_port_set.add(port)

        self.server_address = f"{self.head_ip}:{port}"
        self.service_server = None
        logger.debug(f"Trying to start XLA gRPC server on port: {port}...")
        self.service_server = xla_client._xla.get_distributed_runtime_service(
            self.server_address, self.num_hosts)
        logger.debug(f"Success to start XLA gRPC server on port: {port}...")
        time.sleep(0.5)

        # Launch workers
        self.workers = []
        for i in range(self.num_hosts):
            # Set XLA environment variables
            env_vars = {
                "ALPA_IS_WORKER": "True",
                "NCCL_USE_MULTISTREAM": "False",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": str(
                    global_config.xla_client_mem_fraction),
                "XLA_FLAGS": (
                    os.environ.get("XLA_FLAGS", "") +
                    f" --xla_gpu_autotune_level={global_config.xla_gpu_autotune_level}"
                ),

                # "NCCL_LAUNCH_MODE": "PARALLEL",
                # "XLA_FLAGS": "--xla_dump_to=hlo --xla_dump_hlo_pass_re=.*"
                # "NCCL_DEBUG": "INFO" if i == 0 else "VERSION",
                # "RAY_IGNORE_UNHANDLED_ERRORS": "True",
            }

            if "XLA_PYTHON_CLIENT_ALLOCATOR" in os.environ:
                env_vars["XLA_PYTHON_CLIENT_ALLOCATOR"] = os.environ[
                    "XLA_PYTHON_CLIENT_ALLOCATOR"]

            if "NCCL_LAUNCH_MODE" in os.environ:
                env_vars["NCCL_LAUNCH_MODE"] = os.environ["NCCL_LAUNCH_MODE"]

            if global_config.use_aws_efa:
                env_vars.update({
                    "FI_PROVIDER": "efa",
                    "FI_EFA_USE_DEVICE_RDMA": "1",
                    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH",
                                                      ""),  # For libnccl-net.so
                })

            # Launch a ray actor
            node_resource = "node:" + self.host_info[i]["NodeManagerAddress"]
            cls = ray.remote(num_gpus=self.num_devices_per_host,
                             resources={node_resource: 1e-3})(MeshHostWorker)
            worker = cls.options(runtime_env={
                "env_vars": env_vars
            }).remote(self.server_address, self.num_hosts, i)
            self.workers.append(worker)
        self.sync_workers()
        self.launched = True

    @property
    def host_ips(self):
        ips = [
            self.host_info[i]["NodeManagerAddress"]
            for i, _ in enumerate(self.host_ids)
        ]
        return ips

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    def get_virtual_physical_mesh(self):
        return VirtualPhysicalMesh(
            host_ids=self.host_ids,
            host_info=self.host_info,
            head_ip=self.head_ip,
            num_devices_per_host=self.num_devices_per_host,
            devices=self.devices)

    ##### Buffer Related Functions #####
    def get_remote_buffers(self,
                           buf_refs: List["RemoteBufferRef"],
                           batching=False):
        """Get values of remote buffers."""

        if batching:
            # Batch the remote calls by host ids
            group_by_host_id = [[] for _ in range(self.num_hosts)]
            for buf_ref in buf_refs:
                group_by_host_id[buf_ref.host_id].append(buf_ref.uuid)

            obj_refs = []
            for host_id in range(self.num_hosts):
                obj_refs.append(self.workers[host_id].get_buffers.remote(
                    group_by_host_id[host_id]))

            host_results = ray.get(obj_refs)

            ret = []
            host_cts = [0 for _ in range(self.num_hosts)]
            for buf_ref in buf_refs:
                ret.append(
                    host_results[buf_ref.host_id][host_cts[buf_ref.host_id]])
                host_cts[buf_ref.host_id] += 1

            return ret
        else:
            obj_refs = []
            for buf_ref in buf_refs:
                obj_refs.append(
                    self.workers[buf_ref.host_id].get_buffers.remote(
                        buf_ref.uuid))
            return ray.get(obj_refs)

    def delete_remote_buffers(self, buf_refs: List["RemoteBufferRef"]):
        """Delete remote buffers."""
        if self.workers is None or not ray.is_initialized():
            return

        # Put delete requests into per-host buffers
        for buf_ref in buf_refs:
            self.to_delete_remote_buffers[buf_ref.host_id].append(buf_ref.uuid)
            self.to_delete_remote_buffers_ct = max(
                self.to_delete_remote_buffers_ct,
                len(self.to_delete_remote_buffers[buf_ref.host_id]))

        # Execute the delete requests if there are enough requests
        if self.to_delete_remote_buffers_ct > global_config.delete_remote_buffers_threshold:
            for host_id in range(self.num_hosts):
                self.workers[host_id].delete_buffers.remote(
                    self.to_delete_remote_buffers[host_id])
                self.to_delete_remote_buffers[host_id] = []
            self.to_delete_remote_buffers_ct = 0

    def block_until_ready_remote_buffers(self,
                                         buf_refs: List["RemoteBufferRef"]):
        """Block until the remote buffers are ready."""
        tasks = []
        for buf_ref in buf_refs:
            tasks.append(
                self.workers[buf_ref.host_id].block_until_ready_buffers.remote(
                    buf_ref.uuid))
        ray.get(tasks)

    ##### Executable Related Functions #####
    def shard_batch_arg(self, arg, sharding_spec, num_microbatch, batch_dim,
                        microbatch_aval):
        """Shard high-level arguments as low-level buffers,
        then reorganize them into micro batches."""
        new_spec = get_microbatch_sharding_spec(sharding_spec, batch_dim,
                                                num_microbatch)
        indices = pxla.spec_to_indices(arg.shape, new_spec)
        buf_refs = shard_arg_handlers[type(arg)](arg, self, indices,
                                                 num_microbatch)

        microbatch_arrays = []
        # build distributed array
        for batch_idx in range(num_microbatch):
            refs = [
                buf_refs[device_idx * num_microbatch + batch_idx]
                for device_idx in range(self.num_devices)
            ]
            microbatch_arrays.append(
                DistributedArray(self, microbatch_aval, sharding_spec, refs))
        return microbatch_arrays

    def shard_args_to_bufs(self, shard_indices: Sequence[Sequence[Index]],
                           donated_invars: Sequence[bool], args):
        input_bufs = []
        for arg, indices, donated in zip(args, shard_indices, donated_invars):
            # Fast path for DistributedArray
            if isinstance(arg, DistributedArray) and arg.indices == indices:
                input_bufs.append(arg.remote_buffers)
            elif isinstance(arg, ReplicatedDistributedArray):
                replica = arg.get_replica_on_mesh(self)
                assert replica.indices == indices
                input_bufs.append(replica.remote_buffers)
            else:  # Slow path
                if type(arg) not in [ShapedArray, ShapeDtypeStruct]:
                    arg = xla.canonicalize_dtype(arg)
                buf_refs = shard_arg_handlers[type(arg)](arg, self, indices)
                input_bufs.append(buf_refs)
                if donated and hasattr(arg, "delete"):
                    # shard_arg_handler always creates new buffers,
                    # so we can delete the old buffers
                    arg.delete()

        return input_bufs

    def shard_args_to_arrays(self, avals: Sequence[ShapedArray],
                             shard_indices: Sequence[Sequence[Index]],
                             sharding_specs: Sequence[ShardingSpec], args):
        arrays = []
        for i in range(len(avals)):
            buffers = _shard_array(args[i], self, shard_indices[i])
            #buffers = _device_mesh_put_dummy(args[i], self, shard_indices[i], 1)
            arrays.append(
                DistributedArray(self, avals[i], sharding_specs[i], buffers,
                                 shard_indices[i]))
        return arrays

    def get_outputs_handler(self, avals: Sequence[ShapedArray],
                            sharding_specs: Sequence[ShardingSpec]):
        indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, sharding_specs)
        ]

        def outs_handler(bufs):
            ret = []
            for i, _ in enumerate(avals):
                dis_array = DistributedArray(device_mesh=self,
                                             aval=avals[i],
                                             sharding_spec=sharding_specs[i],
                                             remote_buffers=bufs[i],
                                             indices=indices[i])
                ret.append(dis_array)
            return ret

        return outs_handler

    def delete_remote_executable(self, executable: "MeshDriverExecutable"):
        """Delete remote worker executables of a driver executable."""
        if self.workers is None or not ray.is_initialized():
            return

        for i in range(self.num_hosts):
            self.workers[i].delete_executable.remote(executable.exec_uuid)

    ##### Profiling Related Functions #####
    def profile_hlo_ops(self,
                        op_infos: Sequence[Tuple],
                        cache_filename: str,
                        single_timeout: Optional[float] = None,
                        batch_timeout: Optional[float] = None):
        tasks = []
        for w in self.workers:
            tasks.append(
                w.profile_hlo_ops.remote(op_infos, cache_filename,
                                         single_timeout))
        return ray.get(tasks, timeout=batch_timeout)[0]

    def get_remote_timer(self, timer_name: str):
        return ray.get(self.workers[0].get_timer.remote(timer_name))

    def reset_remote_timer(self, timer_name: str):
        for worker in self.workers:
            ray.get(worker.reset_timer.remote(timer_name))

    def get_memory_allocated(self):
        self.sync_workers()
        return max(
            ray.get([w.get_memory_allocated.remote() for w in self.workers]))

    def get_max_memory_allocated(self):
        self.sync_workers()
        return max(
            ray.get([w.get_max_memory_allocated.remote() for w in self.workers
                    ]))

    def get_available_memory(self):
        return min(
            ray.get([w.get_available_memory.remote() for w in self.workers]))

    def reset_memory_stats(self):
        for worker in self.workers:
            ray.get(worker.reset_memory_stats.remote())

    ##### Other Functions #####
    def sync_workers(self):
        ray.get([w.sync.remote() for w in self.workers])

    def shutdown(self, forced=False):
        if not self.launched:
            return
        if not forced:
            ray.get([w.shutdown.remote() for w in self.workers])
        for worker in self.workers:
            ray.kill(worker)
        self.workers = None
        # shutdown grpc server
        self.service_server.shutdown()
        self.service_server = None
        self.launched = False


class DistributedArray:
    """A distributed array on a PhysicalDeviceMesh."""

    def __init__(self,
                 device_mesh: PhysicalDeviceMesh,
                 aval: ShapedArray,
                 sharding_spec: ShardingSpec,
                 remote_buffers: Sequence["RemoteBufferRef"],
                 indices: Optional[Sequence[Index]] = None):
        self.device_mesh = device_mesh
        self.aval = aval
        self.sharding_spec = sharding_spec
        self.remote_buffers = remote_buffers

        if indices is None:
            indices = pxla.spec_to_indices(self.aval.shape, self.sharding_spec)
        self.indices = indices

        self.shape = self.aval.shape
        self.dtype = self.aval.dtype
        self._npy_value = None
        self._one_replica_buffer_indices = None
        self._fetched_np_buffers = None

    def block_until_ready(self):
        """Block until all remote buffers of this array are ready."""
        self.device_mesh.block_until_ready_remote_buffers(self.remote_buffers)

    def delete(self):
        for buf in self.remote_buffers:
            del buf
        self.device_buffers = None
        self._npy_value = None

    def flush(self):
        self._npy_value = None

    @property
    def one_replica_buffer_indices(self):
        """Indices of buffers containing one complete copy of the array data."""
        if self._one_replica_buffer_indices is None:
            one_replica_indices = []
            seen_index_hashes = set()
            for i, index in enumerate(self.indices):
                hashed_index = _hashable_index(index)
                if hashed_index not in seen_index_hashes:
                    one_replica_indices.append(i)
                    seen_index_hashes.add(hashed_index)
            self._one_replica_buffer_indices = one_replica_indices
        return self._one_replica_buffer_indices

    @property
    def _value(self):
        if self._npy_value is None:
            npy_value = np.empty(self.aval.shape, self.aval.dtype)
            if not self._fetched_np_buffers:
                fetched_np_buffers = self.device_mesh.get_remote_buffers([
                    self.remote_buffers[i]
                    for i in self.one_replica_buffer_indices
                ])
            else:
                fetched_np_buffers = self._fetched_np_buffers
            for ct, i in enumerate(self.one_replica_buffer_indices):
                npy_value[self.indices[i]] = fetched_np_buffers[ct]
            self._npy_value = npy_value
        return self._npy_value

    def __array__(self, dtype=None, context=None):
        return np.asarray(self._value, dtype=dtype)

    def __float__(self):
        return self._value.__float__()

    # TODO(lmzheng): copy more functions from DeviceArray (jax/_src/device_array.py)

    def __str__(self):
        return str(self._value)


def fetch(distributed_arrays: Any):
    """Fetch a pytree of DistributedArray in a batch."""
    buf_refs = []
    device_mesh = distributed_arrays[0].device_mesh

    for array in tree_leaves(distributed_arrays):
        assert array.device_mesh == device_mesh, "Only support fetching from the same mesh."
        for index in array.one_replica_buffer_indices:
            buf_refs.append(array.remote_buffers[index])

    np_arrays = device_mesh.get_remote_buffers(buf_refs, batching=True)

    pt = 0
    for array in distributed_arrays:
        length = len(array.one_replica_buffer_indices)
        array._fetched_np_buffers = np_arrays[pt:pt + length]
        pt += length


core.pytype_aval_mappings[DistributedArray] = attrgetter('aval')
xla.pytype_aval_mappings[DistributedArray] = attrgetter('aval')
xla.canonicalize_dtype_handlers[DistributedArray] = lambda x: x


class ReplicatedDistributedArray:
    """A distributed array that is replicated on multiple meshes.

    We use this class as a workaround for symbols that type-change from DeviceArray
    to DistributedArray in pipeline-parallel training, such as optimizer's step.
    These variables do not have a resharding spec, and cannot be donated, but have a
    replica generates on every participant mesh.

    Warning: do not use this class unless you know exactly how.
    """

    def __init__(self, device_meshes: Sequence[PhysicalDeviceMesh],
                 arrays: Sequence[DistributedArray]):
        self._mesh_array_map = dict()
        self._array_mesh_map = dict()
        for mesh, array in zip(device_meshes, arrays):
            self._mesh_array_map[mesh] = array
            self._array_mesh_map[array] = mesh
        self.aval = self.replica.aval

    def is_replicated_on_mesh(self, mesh):
        """Whether this distributed array is on a given mesh."""
        if mesh in self._mesh_array_map:
            return True
        return False

    def get_replica_on_mesh(self, mesh):
        if not self.is_replicated_on_mesh(mesh):
            raise RuntimeError("No replica found on this mesh.")
        return self._mesh_array_map[mesh]

    def add_replica(self, mesh, array):
        assert isinstance(array, DistributedArray)
        assert isinstance(mesh, PhysicalDeviceMesh)
        if array in self._array_mesh_map:
            raise RuntimeError("Replica exists.")
        if mesh in self._mesh_array_map:
            raise RuntimeError("Mesh exists.")
        self._mesh_array_map.update({mesh: array})
        self._array_mesh_map.update({array: mesh})

    @property
    def replica(self):
        return list(self._mesh_array_map.values())[0]

    @property
    def _value(self):
        return self.replica._value

    def __array__(self, dtype=None, context=None):
        return np.asarray(self._value, dtype=dtype)

    def __str__(self):
        return str(self._value)


core.pytype_aval_mappings[ReplicatedDistributedArray] = attrgetter('aval')
xla.pytype_aval_mappings[ReplicatedDistributedArray] = attrgetter('aval')
xla.canonicalize_dtype_handlers[ReplicatedDistributedArray] = lambda x: x


class VirtualPhysicalMesh:
    """
    A virtual physical mesh used for pipeline parallel compilation.

    VirtualPhysicalMesh is used during compile time. We don't allocate actual workers for it.
    When compilation is finished, we instantiated it as a PhysicalDeviceMesh and launch workers.

    A VirtualPhysicalMesh can also be sliced into multiple VirtualPhysicalMesh.
    """

    def __init__(self,
                 host_ids: Sequence[int] = None,
                 host_info: Sequence[dict] = None,
                 head_ip: str = None,
                 num_devices_per_host: int = 1,
                 devices: Sequence[Sequence[int]] = None):
        self.host_ids = host_ids
        self.host_info = host_info
        self.head_ip = head_ip
        self.num_devices_per_host = num_devices_per_host
        self.is_distributed = True

        if devices is not None:
            if len(devices) != len(host_ids):
                raise RuntimeError(
                    "Please specify the gpu IDs used on each host.")
            if not all(len(ids) == num_devices_per_host for ids in devices):
                raise RuntimeError(
                    "Device IDs specified for each host does not align "
                    "with `num_devices_per_host`.")
        else:
            devices = [list(range(num_devices_per_host)) for _ in host_ids]

        self.devices = devices
        # Depending on gpu_ids, generate device strs and ask Ray to allocate.
        self.device_strs = []
        for i in range(self.num_hosts):
            ip = self.host_info[i]["NodeManagerAddress"]
            self.device_strs.extend(
                [device_id_to_str(ip, j) for j in devices[i]])

    def slice_1d(self, dim, indices):
        """
        Slice a mesh given the slicing config.

        Args:
            dim (int): which dimension to slice from, 0 is host or 1 is the gpu
            indices (List[int]): indices to include along this dimension.

        Returns:
            mesh (PhysicalDeviceMesh)
        """
        if dim == 0:
            # slicing along the host dimension
            host_ids = [self.host_ids[x] for x in indices]
            host_info = [self.host_info[x] for x in host_ids]
            return VirtualPhysicalMesh(
                host_ids=host_ids,
                host_info=host_info,
                head_ip=self.head_ip,
                num_devices_per_host=self.num_devices_per_host)
        else:
            # slicing along the device dimension
            return VirtualPhysicalMesh(host_ids=self.host_ids,
                                       host_info=self.host_info,
                                       head_ip=self.head_ip,
                                       num_devices_per_host=len(indices[0]),
                                       devices=indices)

    def slice_2d(self, host_indices, device_indices):
        host_ids = [self.host_ids[x] for x in host_indices]
        host_info = [self.host_info[x] for x in host_indices]
        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=host_info,
                                   head_ip=self.head_ip,
                                   num_devices_per_host=len(device_indices[0]),
                                   devices=device_indices)

    def slice_profiling_submeshes(self, submesh_num_hosts,
                                  submesh_num_devices_per_host):
        num_hosts = len(self.host_ids)
        num_devices_per_host = self.num_devices_per_host
        num_host_submeshes = num_hosts // submesh_num_hosts
        num_device_submeshes = num_devices_per_host // submesh_num_devices_per_host
        all_submeshes = []
        for i in range(num_host_submeshes):
            for j in range(num_device_submeshes):
                host_indices = range(i * submesh_num_hosts,
                                     (i + 1) * submesh_num_hosts)
                device_indices = [
                    range(j * submesh_num_devices_per_host,
                          (j + 1) * submesh_num_devices_per_host)
                    for _ in host_indices
                ]
                all_submeshes.append(self.slice_2d(host_indices,
                                                   device_indices))
        return all_submeshes

    @property
    def shape(self):
        return (len(self.host_ids), self.num_devices_per_host)

    @property
    def num_devices(self):
        """Return the total number of GPUs on this mesh."""
        return len(self.host_ids) * self.num_devices_per_host

    @property
    def num_hosts(self):
        """Return the number of hosts in the mesh."""
        return len(self.host_ids)

    def get_physical_mesh(self):
        """Convert to a physical mesh (which will request resources from Ray)."""
        return DistributedPhysicalDeviceMesh(
            host_ids=self.host_ids,
            host_info=self.host_info,
            head_ip=self.head_ip,
            num_devices_per_host=self.num_devices_per_host,
            devices=self.devices)

    def get_logical_mesh(self, mesh_shape, mesh_alpha=None, mesh_beta=None):
        """Generate a logical mesh."""
        id_mesh = np.arange(self.num_devices).reshape(mesh_shape)
        mesh_alpha = mesh_alpha or (1.0,) * len(mesh_shape)
        mesh_beta = mesh_beta or (1.0,) * len(mesh_shape)
        return LogicalDeviceMesh(self, id_mesh, mesh_alpha, mesh_beta)

    def get_default_logical_mesh(self):
        """Return the default logical mesh."""
        if self.num_hosts == 1:
            return self.get_logical_mesh(
                (self.num_hosts, self.num_devices_per_host), [1, 1], [1, 1])
        else:
            return self.get_logical_mesh(
                (self.num_hosts, self.num_devices_per_host), [1, 1], [1, 0.1])

    def get_1d_logical_mesh(self):
        """Return a 1D logical mesh."""
        return self.get_logical_mesh((1, self.num_devices))


def set_jax_env_on_driver(use_cpu_on_driver=True):
    """Set jax environment flags for the driver process, so the driver
    process can release GPU memory for the worker processes."""

    # Use cpu backend
    if use_cpu_on_driver:
        jax.config.update("jax_platform_name", "cpu")


class DeviceCluster:
    """A ray cluster with GPU devices."""

    def __init__(self, use_cpu_on_driver=True):
        # pylint: disable=import-outside-toplevel
        from ray.worker import _global_node as ray_global_node
        try:
            self.head_info = ray_global_node.address_info
        except AttributeError:
            raise RuntimeError(
                "Cannot access ray global node. Did you call ray.init?")
        self.head_ip = self.head_info["node_ip_address"]

        # Gather host ids
        self.host_info = []
        for node in ray.nodes():
            for key in node["Resources"]:
                if key.startswith("node:"):
                    self.host_info.append(node)

        # Gather device info
        self.host_num_devices = []
        for host_info in self.host_info:
            number = host_info["Resources"]["GPU"]
            assert number.is_integer()
            self.host_num_devices.append(int(number))

        set_jax_env_on_driver(use_cpu_on_driver)

    @property
    def num_cpus(self):
        return sum(
            map(lambda info: int(info["Resources"]["CPU"]), self.host_info))

    @property
    def num_devices(self):
        return sum(self.host_num_devices)

    def get_physical_mesh(self,
                          host_ids: Sequence[int] = None,
                          num_devices_per_host: int = None):
        """
        Slice a subset of hosts and devices to form a physical device mesh.

        Args:
            host_ids: The index of host nodes.
                'None' means using all hosts
            num_devices_per_host: The number of devices per host.
                'None' means using all devices

        Return:
            A physical multi-host device mesh
        """
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        num_devices_per_host = num_devices_per_host or self.host_num_devices[
            host_ids[0]]
        for host_id in host_ids:
            assert self.host_num_devices[host_id] >= num_devices_per_host

        return DistributedPhysicalDeviceMesh(
            host_ids=host_ids,
            host_info=host_info,
            num_devices_per_host=num_devices_per_host,
            head_ip=self.head_ip)

    def get_virtual_physical_mesh(self,
                                  host_ids: Sequence[int] = None,
                                  num_devices_per_host: int = None):
        """
        Slice a subset of hosts and devices to form a virtual physical mesh.

        The only difference between a virtual and a physical mesh is that a virtual
        mesh does not request cluster resources.
        """
        host_ids = host_ids or np.arange(len(self.host_info))
        host_info = [self.host_info[x] for x in host_ids]

        num_devices_per_host = num_devices_per_host or self.host_num_devices[
            host_ids[0]]
        for host_id in host_ids:
            assert self.host_num_devices[host_id] >= num_devices_per_host

        return VirtualPhysicalMesh(host_ids=host_ids,
                                   host_info=host_info,
                                   num_devices_per_host=num_devices_per_host,
                                   head_ip=self.head_ip)

    def profile_all(self, *args, **kwargs):
        """Profile computation and communication cost for all submesh shapes of this cluster."""
        return mesh_profiling.profile_all(self, *args, **kwargs)


########################################
# Register ShardArg Handler
########################################
def _device_mesh_put(device_mesh, shards, num_batch, batch_dim):
    from alpa.mesh_executable import create_remote_buffer_refs
    buf_refs, buf_uuids = create_remote_buffer_refs(device_mesh, num_batch)
    device_ids = np.arange(device_mesh.num_devices_per_host)
    buf_step = device_mesh.num_devices_per_host * num_batch
    shard_step = device_mesh.num_devices_per_host
    for host_id in range(device_mesh.num_hosts):
        device_mesh.workers[host_id].put_buffers.remote(
            buf_uuids[host_id * buf_step:(host_id + 1) * buf_step], device_ids,
            shards[host_id * shard_step:(host_id + 1) * shard_step], num_batch,
            batch_dim)
    return buf_refs


def _device_mesh_put_dummy(array, device_mesh, indices, num_batch):
    from alpa.mesh_executable import create_remote_buffer_refs
    buf_refs, buf_uuids = create_remote_buffer_refs(device_mesh, num_batch)
    step = device_mesh.num_devices_per_host * num_batch
    for host_id in range(device_mesh.num_hosts):
        device_mesh.workers[host_id].shard_and_put_non_zero_buffer.remote(
            buf_uuids[host_id * step:(host_id + 1) * step], array.shape,
            array.dtype, indices[host_id * step:(host_id + 1) * step],
            num_batch)
    return buf_refs


def _shard_array(array, device_mesh, indices, num_batch=1, batch_dim=0):
    if global_config.use_dummy_value_for_benchmarking:
        return _device_mesh_put_dummy(array, device_mesh, indices, num_batch)
    else:
        # Create shards according to indices for a numpy array
        datas = [array[i] for i in indices]
        if num_batch > 1:
            concate_datas = []
            for device_id in range(device_mesh.num_devices):
                mb = datas[device_id * num_batch:(device_id + 1) * num_batch]
                concate_datas.append(np.concatenate(mb, axis=batch_dim))
            datas = concate_datas
        return _device_mesh_put(device_mesh, datas, num_batch, batch_dim)


def _shard_abstract_array(array,
                          device_mesh,
                          indices,
                          num_batch=1,
                          batch_dim=0):
    assert global_config.use_dummy_value_for_benchmarking is True
    return _device_mesh_put_dummy(array, device_mesh, indices, num_batch)


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def _split_and_concate(array,
                       start_indices,
                       limit_indices,
                       removed_dims,
                       num_batch=1,
                       batch_dim=0):
    shards = _multi_slice(array, start_indices, limit_indices, removed_dims)
    if num_batch > 1:
        concate_datas = []
        step_num = len(shards) // num_batch
        for shard_id in range(step_num):
            concate_datas.append(
                jnp.concatenate(shards[shard_id * num_batch:(shard_id + 1) *
                                       num_batch],
                                axis=batch_dim))
        shards = concate_datas
    return shards


def _shard_device_array(array, device_mesh, indices, num_batch=1, batch_dim=0):
    if global_config.use_dummy_value_for_benchmarking:
        return _device_mesh_put_dummy(array, device_mesh, indices, num_batch)
    else:
        # Create shards according to indices for a DeviceArray
        start_indices, limit_indices, removed_dims = map(
            tuple, unzip3(_as_slice_indices(array, idx) for idx in indices))
        shards = _split_and_concate(array, start_indices, limit_indices,
                                    removed_dims, num_batch, batch_dim)

    return _device_mesh_put(device_mesh, shards, num_batch, batch_dim)


def _shard_distributed_array(array,
                             device_mesh,
                             indices,
                             num_batch=1,
                             batch_dim=0):
    # Slow path: gather values to host and reshard
    return shard_arg_handlers[type(array._value)](array._value, device_mesh,
                                                  indices, num_batch, batch_dim)

# in XLA pred(bool) and uint8 are different, but xla->dlpack->xla
# turns a bool into uint8. This implementation is slow.
def _uint8_to_bool(xla_buffer):
    buf = xla_buffer_to_jax_tensor(xla_buffer).astype(np.bool_)
    return jax_tensor_to_xla_buffer(buf)


shard_arg_handlers = {}  # Shard an argument to a distributed device mesh
for t in array_types:
    shard_arg_handlers[t] = _shard_array
shard_arg_handlers[ShapedArray] = _shard_abstract_array
shard_arg_handlers[ShapeDtypeStruct] = _shard_abstract_array
shard_arg_handlers[xla._DeviceArray] = _shard_device_array
shard_arg_handlers[xla._CppDeviceArray] = _shard_device_array
shard_arg_handlers[DistributedArray] = _shard_distributed_array
shard_arg_handlers[ShardedDeviceArray] = _shard_distributed_array
