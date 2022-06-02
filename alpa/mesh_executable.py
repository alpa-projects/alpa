# pylint: disable=arguments-differ
"""
A mesh executable encapsulates all compiled binary and meta information of a distributed executable.

A mesh executable contains one or several XLA executables.
For each type of mesh executable, there is a driver part and a worker part.
The driver part runs on the user script and the worker parts run on distributed workers.
The driver part sends control commands to launch the worker parts on workers.
"""
from abc import ABC, abstractmethod
import logging
from typing import Callable, Sequence, Optional
import os

import jax.numpy as jnp
from jax._src.lib import xla_bridge as xb, xla_extension as xe
from jax.core import ShapedArray
from jax.interpreters import pxla
from jax.tree_util import tree_flatten, tree_unflatten
import numpy as np
import ray

from alpa.device_mesh import (LocalPhysicalDeviceMesh,
                              DistributedPhysicalDeviceMesh)
from alpa.global_env import global_config
from alpa.measure_record import StrategyConfig
from alpa.shard_parallel.auto_sharding import (get_input_output_sharding_specs,
                                               make_replicated_spec,
                                               run_backend_compilation)
from alpa.timer import timers
from alpa.util import (compile_allocate_zero_buffers,
                       compile_memset_zero_buffers, get_compile_options,
                       get_shard_shape, get_microbatch_sharding_spec,
                       profile_xla_executable)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The global executable and buffer counter.
mesh_executable_counter = 0
remote_buffer_counter = 0


class MeshDriverExecutable(ABC):
    """The base class of the driver part of a mesh executable."""

    @abstractmethod
    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver.

        Args:
            args: The original arguments of the parallelized function.
            kwargs: The additional arguments to control execution options.
        """
        raise NotImplementedError()

    def preshard_dynamic_args(self, *args):
        """Pre-shard the input arguments."""
        raise NotImplementedError()

    def profile_with_dummy_inputs(self, **kwargs):
        """Profile the execution time costs with dummy inputs.

        Args:
            kwargs: The additional arguments to control execution options.
        """
        raise NotImplementedError()

    def get_execution_time_costs(self, warmup: int):
        """Get the execution time costs with internal timers."""
        raise NotImplementedError()

    def get_hlo_text(self):
        """Return the HLO IR in the text format."""
        raise NotImplementedError()

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
        raise NotImplementedError()

    def sync(self):
        """Sync all workers"""
        self.physical_mesh.sync_workers()

    def __del__(self):
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            self.physical_mesh.delete_remote_executable(self.exec_uuid)


class MeshWorkerExecutable(ABC):
    """The base class of the worker part of a mesh executable."""

    @abstractmethod
    def execute_on_worker(self, *arg, **kwargs):
        """Run the executable on the worker."""
        raise NotImplementedError()

    def profile_with_dummy_inputs(self, backend, local_devices):
        """Profile the execution time costs with dummy inputs."""
        raise NotImplementedError()

    def get_hlo_text(self):
        """Return the HLO IR in the text format."""
        raise NotImplementedError()

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
        raise NotImplementedError()


class RemoteBufferRef:
    """A reference to a remote device buffer."""

    def __init__(self,
                 device_mesh,
                 host_id: int,
                 device_id: int,
                 uuid: int = None):
        self.device_mesh = device_mesh
        self.host_id = host_id
        self.device_id = device_id
        self.uuid = uuid if uuid is not None else next_remote_buffer_uuid()
        self.is_deleted_on_workers = False
        logger.debug(f"RemoteBufferRef uuid: {self.uuid} created on mesh "
                     f"with devices {self.device_mesh.device_strs}.")

    def set_deleted_on_workers(self):
        """
        Set the buffer as deleted on workers.

        For some buffers (e.g., donated buffers), if we know the workers has
        already deleted them, then we do not need to do the remote call "delete_remote_buffers"
        again.
        """
        self.is_deleted_on_workers = True

    def __repr__(self):
        return (f"RemoteBufferRef(uuid = {self.uuid}, "
                f"loc = ({self.host_id}, {self.device_id}))")

    def __del__(self):
        if not self.is_deleted_on_workers:
            self.device_mesh.delete_remote_buffers((self,))


def next_mesh_executable_uuid():
    """Return the next uuid of a mesh executable."""
    global mesh_executable_counter
    mesh_executable_counter = (mesh_executable_counter + 1) % (1 << 60)
    return mesh_executable_counter


def next_remote_buffer_uuid(number=1):
    """Return the next uuid of a remote buffer."""
    global remote_buffer_counter
    if number == 1:
        ret = remote_buffer_counter
    else:
        ret = np.arange(remote_buffer_counter, remote_buffer_counter + number)
    remote_buffer_counter = (remote_buffer_counter + number) % (1 << 60)
    return ret


def create_remote_buffer_refs(device_mesh,
                              num_batches=1,
                              host_indices=None,
                              device_indices=None):
    """Create remote buffer references for an distribued array on a device mesh."""
    if host_indices is None:
        host_indices = range(device_mesh.num_hosts)
    if device_indices is None:
        device_indices = range(device_mesh.num_devices_per_host)
    size = len(host_indices) * len(device_indices) * num_batches
    uuids = next_remote_buffer_uuid(size)
    if size == 1:
        uuids = (uuids,)
    uuid_iter = iter(uuids)
    refs = []
    for host_id in host_indices:
        for device_id in device_indices:
            for _ in range(num_batches):
                refs.append(
                    RemoteBufferRef(device_mesh, host_id, device_id,
                                    next(uuid_iter)))
    return refs, uuids


def get_execution_timer_name(exec_uuid: int):
    """Return the name of the timer used for recording pure execution time."""
    return f"{exec_uuid}-execution"


def get_sync_func_driver(physical_mesh):
    """Get the sync function on the driver."""

    def sync_func_driver():
        assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
        physical_mesh.devices[0].synchronize_all_activity()

    return sync_func_driver


def get_sync_func_worker(worker):
    """Get the sync function on the workers"""

    def sync_func_worker():
        worker.local_devices[0].synchronize_all_activity()

    return sync_func_worker


def get_uuid_np_array(array: Sequence[Sequence[int]]):
    """Convert a 2d array of RemoteBufferRef to a np array of UUID (int64)."""
    shape = (len(array), len(array[0]))
    ret = np.empty(shape, dtype=np.int64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[i, j] = array[i][j].uuid
    return ret


class NormalMeshDriverExecutable(MeshDriverExecutable):
    """The driver part of a normal mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 hlo_module: xe.HloModule,
                 strategy_config: StrategyConfig,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 static_argnums: Optional[Sequence[int]] = None,
                 out_tree_thunk: Optional[Callable] = None,
                 flop_count: Optional[int] = None):
        self.physical_mesh = physical_mesh
        self.hlo_module = hlo_module
        self.avals = avals
        self.out_avals = out_avals
        self.donated_invars = donated_invars
        self.static_argnums = static_argnums
        self.out_tree_thunk = out_tree_thunk
        self.flop_count = flop_count
        self.auto_sharding_objective = strategy_config.auto_sharding_objective

        # Read sharding specs
        self.input_sharding_specs, self.output_sharding_specs = get_input_output_sharding_specs(
            hlo_module, avals, out_avals, physical_mesh.num_devices,
            strategy_config.logical_mesh_shape)

        # Cache results for input and output sharding
        self.input_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, self.input_sharding_specs)
        ]
        self.outs_handler = physical_mesh.get_outputs_handler(
            out_avals, self.output_sharding_specs)

        # Send the executable to workers
        self.hlo_text = None
        self.exec_uuid = next_mesh_executable_uuid()
        self._set_executable(physical_mesh, hlo_module, strategy_config)

        # Set up timers
        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def _set_executable(self, physical_mesh, hlo_module, strategy_config):
        """Put the executable on workers."""
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        NormalMeshWorkerExecutable, hlo_proto,
                                        strategy_config)
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)

            backend = xb.get_backend("gpu")

            if physical_mesh.devices[0] is None:
                # A fake physical mesh for generating HLO module only
                self.compiled = run_backend_compilation(
                    backend,
                    hlo_module,
                    strategy_config,
                    physical_mesh.num_devices,
                    bypass_device_assignment_check=True)
            else:
                self.compiled = run_backend_compilation(
                    backend, hlo_module, strategy_config,
                    physical_mesh.num_devices)
            self.hlo_text = self.compiled.hlo_modules()[0].to_string()

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host
        num_outs = len(self.out_avals)

        input_bufs = physical_mesh.shard_args_to_bufs(self.input_indices,
                                                      self.donated_invars,
                                                      (False,) * len(args),
                                                      None, args)
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_uuids = (get_uuid_np_array(input_bufs).reshape(
                len(args), num_hosts, num_devices_per_host).transpose([1, 0,
                                                                       2]))

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = (next_remote_buffer_uuid(
                num_hosts * num_outs * num_devices_per_host).reshape(
                    num_hosts, num_outs, num_devices_per_host))

            if "sync_before" not in kwargs:
                kwargs["sync_before"] = kwargs["sync_after"] = (
                    global_config.shard_parallel_sync_for_timer)

            # Execute the SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, input_uuids[i], output_uuids[i], **kwargs)

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, num_devices)
            output_bufs = np.empty((num_outs, physical_mesh.num_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id,
                        output_uuids[i][host_id][device_id])

            # Mark donated input buffers as already deleted on workers.
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.set_deleted_on_workers()
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            sync_func = (self.sync_func
                         if global_config.shard_parallel_sync_for_timer else None)

            timers(self.timer_name).start(sync_func)
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
            timers(self.timer_name).stop(sync_func)

        return self.outs_handler(output_bufs)

    def preshard_dynamic_args(self, *args):
        """Pre-shard the input arguments."""
        input_bufs = self.physical_mesh.shard_args_to_bufs(
            self.input_indices, self.donated_invars,
            (False,) * len(args), None, args)
        outs_handler = self.physical_mesh.get_outputs_handler(
            self.avals, self.input_sharding_specs)
        return outs_handler(input_bufs)

    def __call__(self, *args):
        """Fast call without signature matching."""
        if self.static_argnums:
            dyn_args = [
                args[i]
                for i in range(len(args))
                if i not in self.static_argnums
            ]
        else:
            dyn_args = args
        args_flat, _ = tree_flatten(dyn_args)
        out = self.launch_on_driver(*args_flat)
        return tree_unflatten(self.out_tree_thunk(), out)

    def profile_with_dummy_inputs(self, **kwargs):
        """Profile the execution time costs with dummy inputs."""
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            tasks = []
            for worker in self.physical_mesh.workers:
                tasks.append(
                    worker.profile_executable_with_dummy_inputs.remote(
                        self.exec_uuid, **kwargs))
            costs = ray.get(tasks)
            for cost_vec in costs:
                if np.inf in cost_vec:
                    return [np.inf] * len(cost_vec)
            costs = np.mean(costs, axis=0)
        else:
            assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
            costs = profile_xla_executable(self.compiled, xb.get_backend("gpu"),
                                           self.physical_mesh.devices)
        return costs

    def get_execution_time_costs(self, warmup):
        """Return the pure execution time costs recorded by an internal timer."""
        return self.physical_mesh.get_remote_timer(
            self.timer_name).costs[warmup:]

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            return (ray.get(self.physical_mesh.workers[0].
                            get_exec_total_allocation_size.remote(
                                self.exec_uuid)))
        else:
            assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
            return self.compiled.total_allocation_size()

    def get_hlo_text(self):
        """Return the HLO IR in the text format."""
        if self.hlo_text is not None:
            return self.hlo_text
        assert isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh)
        self.hlo_text = ray.get(
            self.physical_mesh.workers[0].get_exec_hlo_text.remote(
                self.exec_uuid))
        return self.hlo_text


def get_buffers(buffer_dict, uuids):
    """Get buffers by uuids from the local buffer dictionary."""
    return [buffer_dict[uuid] for uuid in uuids]


def set_buffers(buffer_dict, uuids, buffers):
    """Store buffers to the local buffer dictionary."""
    for uuid, buf in zip(uuids, buffers):
        buffer_dict[uuid] = buf


def delete_donated_buffers(buffer_dict, uuids):
    """Delete the donated buffers from the local buffer dictionary."""
    for i in range(len(uuids)):
        for j in range(len(uuids[i])):
            uuid = uuids[i][j]
            if isinstance(uuid,
                          (np.int64, int)) and buffer_dict[uuid].is_deleted():
                del buffer_dict[uuid]


class NormalMeshWorkerExecutable(MeshWorkerExecutable):
    """The worker part of a normal mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo_proto: bytes,
                 strategy_config: StrategyConfig):
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())

        self.compiled = run_backend_compilation(worker.backend, hlo_proto,
                                                strategy_config, num_devices)
        self.worker = worker

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input buffers from uuids
        input_bufs = [get_buffers(buffer_dict, x) for x in input_uuids]

        # Execute the executable
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        try:
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
        except RuntimeError:
            ray.actor.exit_actor()
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, input_uuids)

    def profile_with_dummy_inputs(self, backend, local_devices):
        """Profile the time cost of this executable with dummy inputs."""
        return profile_xla_executable(self.compiled, backend, local_devices)

    def get_hlo_text(self):
        return self.compiled.hlo_modules()[0].to_string()

    def get_total_allocation_size(self):
        return self.compiled.total_allocation_size()

    def __del__(self):
        self.compiled.delete()


def get_grad_sync_channel_ids(hlo_module: xe.HloModule) -> str:
    """Return the channel ids of all-reduces that are used for gradient synchronization.

    The return value is a string containing all channel ids separated by periods.
    (e.g., ".0.12." means channel id 0 and 12)
    """
    return xe.get_grad_sync_channel_ids(hlo_module)


def get_grad_sync_channel_ids_with_hint(hlo_module: xe.HloModule,
                                        hint: Sequence[int]) -> str:
    """Return the channel ids of all-reduces that are used for gradient synchronization.
    see also get_grad_sync_channel_ids.
    """
    return xe.get_grad_sync_channel_ids(hlo_module, hint)


class GradAccMeshDriverExecutable(MeshDriverExecutable):
    """The driver part of a gradient accumulation mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 accumulate_grad: xe.HloModule,
                 apply_grad: xe.HloModule,
                 strategy_config: StrategyConfig,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 grad_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 batch_invars: Sequence[bool],
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 num_micro_batches: int,
                 flop_count: int = None):
        self.physical_mesh = physical_mesh
        self.avals = avals
        self.out_avals = out_avals
        self.grad_avals = grad_avals
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.num_micro_batches = num_micro_batches
        self.flop_count = flop_count
        self.auto_sharding_objective = strategy_config.auto_sharding_objective

        # Read sharding specs
        logical_mesh_shape = strategy_config.logical_mesh_shape
        accumulate_grad_in_avals = [
            avals[i] for i in accumulate_grad_invar_indices
        ] + grad_avals
        apply_grad_in_avals = \
            [avals[i] for i in apply_grad_invar_indices] + grad_avals
        accumulate_grad_input_sharding_specs, grad_sharding_specs = (
            get_input_output_sharding_specs(accumulate_grad,
                                            accumulate_grad_in_avals,
                                            grad_avals,
                                            physical_mesh.num_devices,
                                            logical_mesh_shape))
        apply_grad_input_sharding_specs, output_sharding_specs = (
            get_input_output_sharding_specs(apply_grad, apply_grad_in_avals,
                                            out_avals,
                                            physical_mesh.num_devices,
                                            logical_mesh_shape))
        num_grads = len(grad_avals)
        assert accumulate_grad_input_sharding_specs[
            -num_grads:] == grad_sharding_specs

        global_arg_sharding_specs = [None] * len(avals)
        for i, idx in enumerate(accumulate_grad_invar_indices):
            global_arg_sharding_specs[
                idx] = accumulate_grad_input_sharding_specs[i]
        for i, idx in enumerate(apply_grad_invar_indices):
            if global_arg_sharding_specs[idx] is None:
                global_arg_sharding_specs[
                    idx] = apply_grad_input_sharding_specs[i]
            else:
                assert global_arg_sharding_specs[
                    idx] == apply_grad_input_sharding_specs[i]
        ## Fill in "Replicated" for remaining undefined args
        for i, spec in enumerate(global_arg_sharding_specs):
            if spec is None:
                global_arg_sharding_specs[i] = (make_replicated_spec(
                    avals[i], logical_mesh_shape))

        # Cache results for input and output sharding
        global_batch_arg_indices = [
            i for i in range(len(avals)) if batch_invars[i]
        ]
        global_arg_shard_indices = []
        for i, aval in enumerate(avals):
            if batch_invars[i] and isinstance(self.physical_mesh,
                                              DistributedPhysicalDeviceMesh):
                # The handling of micro batches is different for
                # distributed device mesh.
                batch_dim = 0
                new_shape = (num_micro_batches * aval.shape[0],) + aval.shape[1:]
                new_spec = get_microbatch_sharding_spec(global_arg_sharding_specs[i],
                                                        batch_dim, num_micro_batches)
                global_arg_shard_indices.append(
                    pxla.spec_to_indices(new_shape, new_spec))
            else:
                global_arg_shard_indices.append(
                    pxla.spec_to_indices(aval.shape, global_arg_sharding_specs[i]))

        accumulate_grad_batch_arg_indices = [
            i for i, j in enumerate(accumulate_grad_invar_indices)
            if batch_invars[j]
        ]
        grad_shard_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        grad_shard_dtypes = [aval.dtype for aval in grad_avals]
        self.global_batch_arg_indices = global_batch_arg_indices
        self.global_arg_shard_indices = global_arg_shard_indices
        self.outs_handler = physical_mesh.get_outputs_handler(
            out_avals, output_sharding_specs)

        # Send the executable to workers
        self.exec_uuid = next_mesh_executable_uuid()
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            for w in physical_mesh.workers:
                w.put_executable.remote(
                    self.exec_uuid, GradAccMeshWorkerExecutable,
                    accumulate_grad.as_serialized_hlo_module_proto(),
                    apply_grad.as_serialized_hlo_module_proto(),
                    accumulate_grad_invar_indices, apply_grad_invar_indices,
                    accumulate_grad_batch_arg_indices, grad_shard_shapes,
                    grad_shard_dtypes, strategy_config, donated_invars,
                    batch_invars, num_grads, num_micro_batches)
            self.hlo_text = None  # will be fetched from the workers later
            self.grad_sync_channel_ids = None
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            backend = xb.get_backend("gpu")

            self.accumulate_grad = run_backend_compilation(
                backend, accumulate_grad, strategy_config,
                physical_mesh.num_devices)
            self.apply_grad = run_backend_compilation(backend, apply_grad,
                                                      strategy_config,
                                                      physical_mesh.num_devices)
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                backend, physical_mesh.num_devices, grad_shard_shapes,
                grad_shard_dtypes)
            self.accumulate_grad_batch_arg_indices = accumulate_grad_batch_arg_indices

            self.hlo_text = (self.accumulate_grad.hlo_modules()[0].to_string() +
                             self.apply_grad.hlo_modules()[0].to_string())
            self.grad_sync_channel_ids = get_grad_sync_channel_ids(
                self.accumulate_grad.hlo_modules()[0])
            self.skip_allreduce_env_name = (
                self.accumulate_grad.hlo_modules()[0].name() +
                "XLA_SKIP_NCCL_COLLECTIVE_IDS")

        # Set up timers
        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        num_micro_batches = self.num_micro_batches
        grad_avals = self.grad_avals
        num_grads = len(grad_avals)
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host
        num_outs = len(self.out_avals)

        input_bufs = physical_mesh.shard_args_to_bufs(
            self.global_arg_shard_indices,
            self.donated_invars, self.batch_invars,
            num_micro_batches, args)

        first_batch_bufs = input_bufs
        next_batches_bufs = []
        for i in self.global_batch_arg_indices:
            micro_batches = input_bufs[i]
            first_batch_bufs[i] = micro_batches[0]
            next_batches_bufs.extend(micro_batches[1:])

        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            # Shape: (num_hosts, num_args, num_devices_per_host)
            first_batch_uuids = (get_uuid_np_array(first_batch_bufs).reshape(
                len(first_batch_bufs), num_hosts,
                num_devices_per_host).transpose([1, 0, 2]))

            if next_batches_bufs:
                next_batches_uuids = (get_uuid_np_array(next_batches_bufs).reshape(
                    len(next_batches_bufs), num_hosts,
                    num_devices_per_host).transpose([1, 0, 2]))
            else:
                next_batches_uuids = (None,) * num_hosts

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = (next_remote_buffer_uuid(
                num_hosts * num_outs * num_devices_per_host).reshape(
                    num_hosts, num_outs, num_devices_per_host))

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, first_batch_uuids[i], next_batches_uuids[i],
                    output_uuids[i],
                    global_config.shard_parallel_sync_for_timer,
                    global_config.shard_parallel_sync_for_timer)

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, num_devices)
            output_bufs = np.empty((num_outs, physical_mesh.num_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id,
                        output_uuids[i][host_id][device_id])

            # Mark donated input buffers as already deleted on workers.
            for bufs, is_donated in zip(first_batch_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.set_deleted_on_workers()

            # Mark micro batch buffers as already deleted on workers.
            for bufs in next_batches_bufs:
                for buf in bufs:
                    buf.set_deleted_on_workers()
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            sync_func = (self.sync_func
                         if global_config.shard_parallel_sync_for_timer else None)

            # Prepare gradient buffers
            timers(self.timer_name).start(sync_func)
            grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
                [])

            # Call accumulate_grad multiple times
            tmp_input_bufs = (
                [first_batch_bufs[i] for i in self.accumulate_grad_invar_indices] +
                grad_bufs)
            os.environ[
                self.skip_allreduce_env_name] = self.grad_sync_channel_ids
            for i in range(num_micro_batches):
                if i != 0:
                    # Feed in the data of the next batch
                    tmp_input_bufs[-num_grads:] = grad_bufs
                    for j, idx in enumerate(
                            self.accumulate_grad_batch_arg_indices):
                        tmp_input_bufs[idx] = next_batches_bufs[
                            j * (num_micro_batches - 1) + (i - 1)]
                if i == num_micro_batches - 1:
                    os.environ[self.skip_allreduce_env_name] = ""
                grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                    tmp_input_bufs)

            # Call apply_grad
            tmp_input_bufs = (
                [first_batch_bufs[i] for i in self.apply_grad_invar_indices] +
                grad_bufs)
            output_bufs = self.apply_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)
            timers(self.timer_name).stop(sync_func)

        # Wrap output buffers as ShardedArray
        return self.outs_handler(output_bufs)

    def get_execution_time_costs(self, warmup):
        """Return the pure execution time costs recorded by an internal timer."""
        return self.physical_mesh.get_remote_timer(
            self.timer_name).costs[warmup:]

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            return ray.get(self.physical_mesh.workers[0].
                           get_exec_total_allocation_size.remote(
                               self.exec_uuid))
        else:
            assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
            return max(self.accumulate_grad.total_allocation_size(),
                       self.apply_grad.total_allocation_size())

    def get_hlo_text(self):
        """Return the HLO IR in the text format."""
        if self.hlo_text is not None:
            return self.hlo_text
        assert isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh)
        self.hlo_text = ray.get(
            self.physical_mesh.workers[0].get_exec_hlo_text.remote(
                self.exec_uuid))
        self.grad_sync_channel_ids = ray.get(
            self.physical_mesh.workers[0].get_exec_grad_sync_channel_ids.remote(
                self.exec_uuid))
        return self.hlo_text


class GradAccMeshWorkerExecutable(MeshWorkerExecutable):
    """The worker part of a gradient accumulation mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 accumulate_grad_proto: bytes, apply_grad_proto: bytes,
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 accumulate_grad_batch_arg_indices: Sequence[int],
                 grad_shard_shapes: Sequence[Sequence[int]],
                 grad_shard_dtypes: Sequence[jnp.dtype],
                 strategy_config: StrategyConfig,
                 donated_invars: Sequence[bool], batch_invars: Sequence[bool],
                 num_grads: int, num_micro_batches: int):
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())

        self.accumulate_grad = run_backend_compilation(worker.backend,
                                                       accumulate_grad_proto,
                                                       strategy_config,
                                                       num_devices)
        self.apply_grad = run_backend_compilation(worker.backend,
                                                  apply_grad_proto,
                                                  strategy_config, num_devices)
        self.allocate_zero_buffers = compile_allocate_zero_buffers(
            worker.backend, num_devices, grad_shard_shapes, grad_shard_dtypes)
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.accumulate_grad_batch_arg_indices = accumulate_grad_batch_arg_indices
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.num_grads = num_grads
        self.num_micro_batches = num_micro_batches
        self.buffer_dict = worker.buffers
        self.grad_sync_channel_ids = get_grad_sync_channel_ids(
            self.accumulate_grad.hlo_modules()[0])
        self.skip_allreduce_env_name = (
            self.accumulate_grad.hlo_modules()[0].name() +
            "XLA_SKIP_NCCL_COLLECTIVE_IDS")

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          first_batch_uuids: Sequence[Sequence[int]],
                          next_batches_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.buffer_dict
        num_micro_batches = self.num_micro_batches

        tmp_input_bufs = [
            get_buffers(buffer_dict, first_batch_uuids[i])
            for i in self.accumulate_grad_invar_indices
        ]

        # Prepare gradient buffers
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
            [])

        # Call accumulate_grad multiple times
        tmp_input_bufs = tmp_input_bufs + grad_bufs
        os.environ[self.skip_allreduce_env_name] = self.grad_sync_channel_ids
        for i in range(num_micro_batches):
            if i != 0:
                # Feed in the data of the next batch
                tmp_input_bufs[-self.num_grads:] = grad_bufs
                for j, idx in enumerate(self.accumulate_grad_batch_arg_indices):
                    tmp_input_bufs[idx] = get_buffers(
                        buffer_dict,
                        next_batches_uuids[j * (num_micro_batches - 1) + (i - 1)])
            if i == num_micro_batches - 1:
                os.environ[self.skip_allreduce_env_name] = ""
            grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)

        # Call apply_grad
        tmp_input_bufs = [
            get_buffers(buffer_dict, first_batch_uuids[i])
            for i in self.apply_grad_invar_indices
        ] + grad_bufs
        output_bufs = self.apply_grad.execute_sharded_on_local_devices(
            tmp_input_bufs)
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, first_batch_uuids)

        # Delete micro batch buffers
        if next_batches_uuids is not None:
            for i in range(len(next_batches_uuids)):
                for j in range(len(next_batches_uuids[i])):
                    del buffer_dict[next_batches_uuids[i][j]]

    def get_hlo_text(self):
        return (self.accumulate_grad.hlo_modules()[0].to_string() +
                self.apply_grad.hlo_modules()[0].to_string())

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        return max(self.accumulate_grad.total_allocation_size(),
                   self.apply_grad.total_allocation_size())

    def __del__(self):
        self.accumulate_grad.delete()
        self.apply_grad.delete()
        self.allocate_zero_buffers.delete()


class PartialGradAccMeshDriverExecutable(NormalMeshDriverExecutable):
    """
    The driver part of a mesh executable that can optionally skip
    the gradient synchronization step.

    This executable is used for computation stages in pipeline,
    such as forward, backward and apply_grad
    """

    def __init__(self, physical_mesh: "PhysicalDeviceMesh",
                 hlo_module: xe.HloModule, strategy_config: StrategyConfig,
                 avals: Sequence[ShapedArray], out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 out_acc_grad_indices: Sequence[int]):
        self.out_acc_grad_indices = out_acc_grad_indices

        super().__init__(physical_mesh, hlo_module, strategy_config, avals,
                         out_avals, donated_invars)

    def _set_executable(self, physical_mesh, hlo_module, strategy_config):
        """Put the executable on workers."""
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        PartialGradAccMeshWorkerExecutable,
                                        hlo_proto, strategy_config,
                                        self.out_acc_grad_indices)
            self.hlo_text = None  # will be fetched from the workers later
            self.grad_sync_channel_ids = None
            self.skip_allreduce_env_name = None
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            backend = xb.get_backend("gpu")
            self.compiled = run_backend_compilation(backend, hlo_module,
                                                    strategy_config,
                                                    physical_mesh.num_devices)
            self.hlo_text = self.compiled.hlo_modules()[0].to_string()
            self.grad_sync_channel_ids = get_grad_sync_channel_ids_with_hint(
                self.compiled.hlo_modules()[0], self.out_acc_grad_indices)
            self.skip_allreduce_env_name = (
                self.compiled.hlo_modules()[0].name() +
                "XLA_SKIP_NCCL_COLLECTIVE_IDS")

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        assert 'skip_grad_sync' in kwargs, (
            'Partial grad acc mesh executable missing kwargs "skip_grad_sync"')
        skip_grad_sync = kwargs["skip_grad_sync"]
        os.environ[self.skip_allreduce_env_name] = (self.grad_sync_channel_ids
                                                    if skip_grad_sync else "")
        return super().launch_on_driver(*args, **kwargs)


class PartialGradAccMeshWorkerExecutable(NormalMeshWorkerExecutable):
    """
    The worker part of a mesh executable that can optionally skip
    the gradient synchronization step.

    This executable is used for computation stages in pipeline,
    such as forward, backward and apply_grad
    """

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo_proto: bytes,
                 strategy_config: StrategyConfig, output_acc_grad_indices: str):
        super().__init__(worker, uuid, hlo_proto, strategy_config)
        self.grad_sync_channel_ids = get_grad_sync_channel_ids_with_hint(
            self.compiled.hlo_modules()[0], output_acc_grad_indices)
        self.skip_allreduce_env_name = (self.compiled.hlo_modules()[0].name() +
                                        "XLA_SKIP_NCCL_COLLECTIVE_IDS")

    # pylint: disable=arguments-differ
    def execute_on_worker(self,
                          input_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool,
                          skip_grad_sync: bool):
        """Run the executable on the worker."""
        os.environ[self.skip_allreduce_env_name] = (self.grad_sync_channel_ids
                                                    if skip_grad_sync else "")
        return super().execute_on_worker(input_uuids, output_uuids,
                                         sync_before, sync_after)

    def profile_with_dummy_inputs(self, backend, local_devices, skip_grad_sync):
        """Profile the time cost of this executable with dummy inputs."""
        os.environ[self.skip_allreduce_env_name] = (self.grad_sync_channel_ids
                                                    if skip_grad_sync else "")
        return profile_xla_executable(self.compiled, backend, local_devices)


class AllocZeroBufferDriverExecutable(MeshDriverExecutable):
    """The driver part of a buffer-allocation executable."""

    def __init__(self, physical_mesh: "PhysicalDeviceMesh",
                 grad_vars: Sequence[ShapedArray],
                 grad_sharding_specs: Sequence[pxla.ShardingSpec]):
        self.physical_mesh = physical_mesh
        grad_avals = [var.aval for var in grad_vars]
        grad_shard_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        grad_shard_dtypes = [aval.dtype for aval in grad_avals]
        self.out_avals = grad_avals
        self.outs_handler = physical_mesh.get_outputs_handler(
            grad_avals, grad_sharding_specs)

        self.exec_uuid = next_mesh_executable_uuid()
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        AllocZeroBufferWorkerExecutable,
                                        grad_shard_shapes, grad_shard_dtypes)
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                xb.get_backend("gpu"), physical_mesh.devices, grad_shard_shapes,
                grad_shard_dtypes)

        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        assert len(args) == 0, (
            f"allocate zero buffers does not need args, got {len(args)}")
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_outs = len(self.out_avals)
        num_devices_per_host = physical_mesh.num_devices_per_host

        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            # Get output uuids
            output_uuids = (next_remote_buffer_uuid(
                num_hosts * num_outs * num_devices_per_host).reshape(
                    num_hosts, num_outs, num_devices_per_host))

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, [], output_uuids[i])

            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather outputs
            output_bufs = np.empty((num_outs, physical_mesh.num_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id,
                        output_uuids[i][host_id][device_id])
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            timers(self.timer_name).start(self.sync_func)
            output_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
                [])
            timers(self.timer_name).stop(self.sync_func)

        return self.outs_handler(output_bufs)


class AllocZeroBufferWorkerExecutable(MeshWorkerExecutable):
    """The worker part of a buffer-allocation executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 grad_shard_shapes: Sequence[Sequence[int]],
                 grad_shard_dtypes: Sequence[jnp.dtype]):
        num_devices = len(worker.backend.devices())
        self.allocate_zero_buffers = compile_allocate_zero_buffers(
            worker.backend, num_devices, grad_shard_shapes, grad_shard_dtypes)
        self.worker = worker

        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Execute
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        output_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
            [])
        timers(self.timer_name).stop(self.sync_func if sync_after else None)
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

    def __del__(self):
        self.allocate_zero_buffers.delete()


class MemzeroWorkerExecutable(MeshWorkerExecutable):
    """The worker part of an executable that sets all input tensors to zeros."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 buffer_shard_shapes: Sequence[Sequence[int]],
                 buffer_shard_dtypes: Sequence[jnp.dtype]):
        num_devices = len(worker.backend.devices())
        self.memzero = compile_memset_zero_buffers(worker.backend, num_devices,
                                                   buffer_shard_shapes,
                                                   buffer_shard_dtypes)
        self.worker = worker

        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input
        input_bufs = [get_buffers(buffer_dict, x) for x in input_uuids]

        # Execute
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        _ = self.memzero.execute_sharded_on_local_devices(input_bufs)
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

    def __del__(self):
        self.memzero.delete()


class ConcatMeshWorkerExecutable(MeshWorkerExecutable):
    # This is only a patch. It will be deprecated when we move concat into apply_grad
    def __init__(self, worker, uuid, hlo_proto):
        num_devices = len(worker.backend.devices())
        compile_options = get_compile_options(
            num_replicas=1,
            num_partitions=num_devices,
            device_assignment=np.arange(num_devices).reshape((1, -1)),
            use_spmd_partitioning=False,
            parameter_is_tupled_arguments=False,
            build_random_seed=global_config.build_random_seed)
        xla_computation = xe.XlaComputation(hlo_proto)

        self.concat = worker.backend.compile(xla_computation, compile_options)

        self.worker = worker
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids: Sequence[Sequence[int]],
                          output_uuids: Sequence[Sequence[int]],
                          sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input
        input_bufs = [get_buffers(buffer_dict, x) for x in input_uuids]

        # Execute
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        output_bufs = self.concat.execute_sharded_on_local_devices(input_bufs)
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

    def __del__(self):
        self.concat.delete()
