# pylint: disable=import-outside-toplevel
"""
A mesh executable encapsulates all compiled binary and meta information of a distributed executable.

A mesh executable contains one or several XLA executables.
For each type of mesh executable, there is a driver part and a worker part.
"""
from collections import namedtuple
import logging
import os
from typing import List, Sequence, Tuple, Optional

import numpy as np
import ray

from jax._src.util import partial
from jax.core import ShapedArray
from jax.interpreters import pxla
from jax.interpreters.xla import XlaExecutable
from jax.lib import xla_client, xla_bridge, xla_extension
import jax.numpy as jnp

from parax.measure_record import StrategyConfig
from parax.timer import timers
from parax.util import (compile_allocate_zero_buffers,
                        compile_memset_zero_buffers, get_shard_shape,
                        profile_xla_executable)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# The global executable and buffer counter.
mesh_executable_counter = 0
remote_buffer_counter = 0

ProtoAndSharding = namedtuple("HloProtoAndSharding",
                              ["proto", "input_shardings", "output_shardings"])


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


class RemoteBufferRef:
    """A reference to a remote device buffer."""

    def __init__(self,
                 device_mesh,
                 host_id,
                 device_id,
                 uuid=None,
                 dtype=jnp.float32):
        self.device_mesh = device_mesh
        self.host_id = host_id
        self.device_id = device_id
        self.uuid = uuid if uuid is not None else next_remote_buffer_uuid()
        self.dtype = dtype
        self.is_deleted_on_workers = False
        logger.debug(
            "RemoteBufferRef uuid: {} created on mesh with devices {}.".format(
                self.uuid, self.device_mesh.device_strs))

    def set_deleted_on_workers(self):
        """
        Set the buffer as deleted on workers.

        For some buffers (e.g., donated buffers), if we know the workers has
        already deleted them, then we do not need to do the RPC call "delete_remote_buffers"
        again.
        """
        self.is_deleted_on_workers = True

    def __repr__(self):
        return f"RemoteBufferRef(uuid = {self.uuid}, " \
               f"loc = ({self.host_id}, {self.device_id})), " \
               f"dtype=({self.dtype})"

    def __del__(self):
        if not self.is_deleted_on_workers:
            self.device_mesh.delete_remote_buffers((self,))


def get_uuid_np_array(array):
    """Convert a 2d array of RemoteBufferRef to a np array of UUID (int64)."""
    shape = (len(array), len(array[0]))
    ret = np.empty(shape, dtype=np.int64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[i, j] = array[i][j].uuid
    return ret


class MeshDriverExecutable:
    """The base class of the driver part of a mesh executable."""


class MeshWorkerExecutable:
    """The base class of the worker part of a mesh executable."""


def get_execution_timer_name(exec_uuid):
    """Return the name of the timer used for recording pure execution time."""
    return f"{exec_uuid}-execution"


def get_sync_func_driver(physical_mesh):

    def sync_func_driver():
        physical_mesh.devices[0].synchronize_all_activity()

    return sync_func_driver


def get_sync_func_worker(worker):

    def sync_func_worker():
        worker.local_devices[0].synchronize_all_activity()

    return sync_func_worker


class NormalMeshDriverExecutable(MeshDriverExecutable):
    """The driver part of a normal mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 compiled: XlaExecutable,
                 strategy_config: StrategyConfig,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 flop_count: Optional[int] = None):
        from parax.shard_parallel.auto_sharding import (
            get_input_output_sharding_specs, sharding_proto_to_sharding_spec,
            make_replicated_spec)

        self.physical_mesh = physical_mesh
        self.avals = avals
        self.out_avals = out_avals
        self.donated_invars = donated_invars
        self.flop_count = flop_count

        # Read sharding specs
        if isinstance(compiled, ProtoAndSharding):
            proto = compiled.proto
            input_sharding_protos = compiled.input_shardings
            output_sharding_protos = compiled.output_shardings
            self.hlo_module = xla_client.XlaComputation(proto).as_hlo_module()
            logical_mesh_shape = strategy_config.logical_mesh_shape
            if physical_mesh.total_devices != 1:
                self.input_sharding_specs = [
                    sharding_proto_to_sharding_spec(proto_tuple, aval,
                                                    logical_mesh_shape)
                    for (proto_tuple, aval) in zip(input_sharding_protos, avals)
                ]
                self.output_sharding_specs = sharding_proto_to_sharding_spec(
                    output_sharding_protos, out_avals, logical_mesh_shape)
            else:
                self.input_sharding_specs = [
                    make_replicated_spec(aval, logical_mesh_shape)
                    for aval in avals
                ]
                self.output_sharding_specs = [
                    make_replicated_spec(aval, logical_mesh_shape)
                    for aval in out_avals
                ]
        else:
            self.hlo_module = compiled.hlo_modules()[0]
            self.input_sharding_specs, self.output_sharding_specs = get_input_output_sharding_specs(
                self.hlo_module, physical_mesh.total_devices, avals, out_avals,
                strategy_config.logical_mesh_shape)

        # Cache results for input and output sharding
        self.input_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, self.input_sharding_specs)
        ]
        self.outs_handler = physical_mesh.get_outputs_handler(
            out_avals, self.output_sharding_specs)

        # Send the executable to workers
        self.exec_uuid = next_mesh_executable_uuid()
        self.hlo_text = self.hlo_module.to_string()

        self.set_executable(physical_mesh, compiled, strategy_config)

        # Set up timers
        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def set_executable(self, physical_mesh, compiled, strategy_config):
        if physical_mesh.is_distributed:
            hlo_proto = self.hlo_module.as_serialized_hlo_module_proto()
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        NormalMeshWorkerExecutable, hlo_proto,
                                        strategy_config)
        else:
            self.compiled = compiled

    def get_driver_callable(self):
        """Get a callable that runs on the driver and handles arguments/outputs conversion."""
        ret = partial(self.launch_on_driver)
        ret.shard_args_only = partial(self.shard_args_only)
        ret.get_executable = lambda: self
        return ret

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host
        num_outs = len(self.out_avals)

        input_bufs = physical_mesh.shard_args(self.input_indices,
                                              self.donated_invars, args)
        if physical_mesh.is_distributed:
            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_uuids = get_uuid_np_array(input_bufs)\
                .reshape(len(args), num_hosts, num_devices_per_host)\
                .transpose([1, 0, 2])

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = next_remote_buffer_uuid(num_hosts * num_outs * num_devices_per_host)\
                .reshape(num_hosts, num_outs, num_devices_per_host)\

            # Execute the SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, input_uuids[i], output_uuids[i], **kwargs)

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, total_devices)
            output_bufs = np.empty((num_outs, physical_mesh.total_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh,
                        host_id,
                        device_id,
                        output_uuids[i][host_id][device_id],
                        dtype=self.out_avals[i].dtype)

            # Mark donated input buffers as already deleted on workers.
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.set_deleted_on_workers()
        else:
            timers(self.timer_name).start(self.sync_func)
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
            timers(self.timer_name).stop(self.sync_func)

        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        input_bufs = self.physical_mesh.shard_args(self.input_indices,
                                                   self.donated_invars, args)
        outs_handler = self.physical_mesh.get_outputs_handler(
            self.avals, self.input_sharding_specs)
        return outs_handler(input_bufs)

    def profile_with_dummy_inputs(self, **kwargs):
        """Profile the time cost of this executable with dummy inputs."""
        if self.physical_mesh.is_distributed:
            tasks = []
            for worker in self.physical_mesh.workers:
                tasks.append(
                    worker.profile_executable_with_dummy_inputs.remote(
                        self.exec_uuid, **kwargs))
            costs = ray.get(tasks)
            for cost_vec in costs:
                if np.inf in cost_vec:
                    return [np.inf] * len(cost_vec)
            costs = costs[0]
        else:
            costs = profile_xla_executable(self.compiled,
                                           xla_bridge.get_backend("gpu"),
                                           self.physical_mesh.devices)
        return costs

    def get_execution_time_costs(self, warmup):
        """Return the pure execution time costs recorded by an internal timer."""
        return self.physical_mesh.get_remote_timer(
            self.timer_name).costs[warmup:]

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        if self.physical_mesh.is_distributed:
            return ray.get(self.physical_mesh.workers[0].\
                get_exec_total_allocation_size.remote(self.exec_uuid))
        else:
            return self.compiled.total_allocation_size()

    def get_hlo_text(self):
        return self.hlo_text

    def __del__(self):
        self.physical_mesh.delete_remote_executable(self)


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


class NormalMeshWorkerExecutable:
    """The worker part of a normal mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo_proto: bytes,
                 strategy_config: StrategyConfig):
        from parax.shard_parallel.auto_sharding import (
            compile_with_given_strategy, HloProtoStatus)

        xla_computation = xla_client.XlaComputation(hlo_proto)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())
        hlo_proto_status = HloProtoStatus.FULLY_OPTIMIZED

        self.compiled = compile_with_given_strategy(worker.backend,
                                                    xla_computation,
                                                    strategy_config,
                                                    num_devices, False,
                                                    hlo_proto_status)
        self.worker = worker

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids: List[List[int]],
                          output_uuids: List[List[int]],
                          sync_before=True,
                          sync_after=True):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input buffers from uuids
        input_bufs = [get_buffers(buffer_dict, x) for x in input_uuids]

        before_sync_func = self.sync_func if sync_before else None
        after_sync_func = self.sync_func if sync_after else None
        # Execute the executable
        timers(self.timer_name).start(before_sync_func)

        # TODO(Hao): try has an overhead. Is there better ways?
        # output_bufs = self.compiled.execute_sharded_on_local_devices(input_bufs)
        try:
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
        except RuntimeError as re:
            # logger.info("Executing in actor encounters an exception: {}".format(re))
            ray.actor.exit_actor()

        timers(self.timer_name).stop(after_sync_func)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, input_uuids)

    def profile_with_dummy_inputs(self, backend, local_devices, **kwargs):
        """Profile the time cost of this executable with dummy inputs."""
        if len(kwargs):
            logger.warning(f"kwargs {(list(kwargs.keys()))} are ignored")
        return profile_xla_executable(self.compiled, backend, local_devices)

    def get_total_allocation_size(self):
        return self.compiled.total_allocation_size()

    def __del__(self):
        self.compiled.delete()


def get_grad_sync_channel_ids(hlo_module) -> str:
    """Return the channel ids of all-reduces that are used for gradient synchronization.

    The return value is a string containing all channel ids separated by periods.
    (e.g., ".0.12." means channel id 0 and 12)
    """
    return xla_extension.get_grad_sync_channel_ids(hlo_module)
    #hlo_ir = hlo_module.to_string()
    #import re
    #ids = []
    #for item in re.findall("all-reduce\(.*channel_id=(.*), replica_groups=\{\{0,4", hlo_ir):
    ##for item in re.findall("all-reduce\(.*channel_id=(.*), replica_groups=", hlo_ir):
    #    ids.append(item)

    #ids = "." + ".".join(ids) + "."
    #return ids


class GradAccMeshDriverExecutable:
    """The driver part of a gradient accumulation mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 accumulate_grad: XlaExecutable,
                 apply_grad: XlaExecutable,
                 strategy_config: StrategyConfig,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 grad_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 batch_invars: Sequence[bool],
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 num_micro_batches: int,
                 flop_count: Optional[int] = None):
        from parax.shard_parallel.auto_sharding import (
            get_input_output_sharding_specs, make_replicated_spec)

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

        # Read sharding specs
        logical_mesh_shape = strategy_config.logical_mesh_shape
        accumulate_grad_in_avals = [
            avals[i] for i in accumulate_grad_invar_indices
        ] + grad_avals
        apply_grad_in_avals = [avals[i] for i in apply_grad_invar_indices
                              ] + grad_avals
        accumulate_grad_input_sharding_specs, grad_sharding_specs =\
            get_input_output_sharding_specs(
            accumulate_grad.hlo_modules()[0], physical_mesh.total_devices,
            accumulate_grad_in_avals, grad_avals, logical_mesh_shape)
        apply_grad_input_sharding_specs, output_sharding_specs =\
            get_input_output_sharding_specs(
            apply_grad.hlo_modules()[0], physical_mesh.total_devices,
            apply_grad_in_avals, out_avals, logical_mesh_shape)
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
                global_arg_sharding_specs[i] =\
                    make_replicated_spec(avals[i], logical_mesh_shape)

        # Get the channel ids of gradient sync all-reduce
        grad_sync_channel_ids =\
            get_grad_sync_channel_ids(accumulate_grad.hlo_modules()[0])

        # Cache results for input and output sharding
        global_arg_shard_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, global_arg_sharding_specs)
        ]
        global_batch_arg_indices = [
            i for i in range(len(avals)) if batch_invars[i]
        ]
        accumulate_grad_batch_arg_indices = [
            i for i, j in enumerate(accumulate_grad_invar_indices)
            if batch_invars[j]
        ]
        next_batch_indices = []
        for i in global_batch_arg_indices:
            next_batch_indices.extend([global_arg_shard_indices[i]] *
                                      (num_micro_batches - 1))
        grad_shard_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        grad_shard_dtypes = [aval.dtype for aval in grad_avals]
        self.outs_handler = physical_mesh.get_outputs_handler(
            out_avals, output_sharding_specs)
        self.global_batch_arg_indices = global_batch_arg_indices
        self.next_batch_indices = next_batch_indices
        self.global_arg_shard_indices = global_arg_shard_indices

        # Send the executable to workers
        self.exec_uuid = next_mesh_executable_uuid()
        self.hlo_text = accumulate_grad.hlo_modules()[0].to_string() +\
                        apply_grad.hlo_modules()[0].to_string()
        if physical_mesh.is_distributed:
            for w in physical_mesh.workers:
                w.put_executable.remote(
                    self.exec_uuid, GradAccMeshWorkerExecutable,
                    accumulate_grad.hlo_modules()\
                        [0].as_serialized_hlo_module_proto(),
                    apply_grad.hlo_modules()\
                        [0].as_serialized_hlo_module_proto(),
                    accumulate_grad_invar_indices, apply_grad_invar_indices,
                    accumulate_grad_batch_arg_indices, grad_shard_shapes,
                    grad_shard_dtypes, strategy_config, donated_invars,
                    batch_invars, num_grads, num_micro_batches,
                    grad_sync_channel_ids)
            self.grad_sync_channel_ids = grad_sync_channel_ids
        else:
            self.accumulate_grad = accumulate_grad
            self.apply_grad = apply_grad
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                xla_bridge.get_backend("gpu"), physical_mesh.total_devices,
                grad_shard_shapes, grad_shard_dtypes)
            self.accumulate_grad_batch_arg_indices = accumulate_grad_batch_arg_indices
            self.grad_sync_channel_ids = grad_sync_channel_ids
            self.skip_allreduce_env_name =\
                self.accumulate_grad.hlo_modules()[0].name() +\
                "XLA_SKIP_NCCL_COLLECTIVE_IDS"

        # Set up timers
        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def get_driver_callable(self):
        """Get a callable that runs on the driver and handles arguments/outputs conversion."""
        ret = partial(self.launch_on_driver)
        ret.shard_args_only = partial(self.shard_args_only)
        ret.get_executable = lambda: self
        return ret

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        num_micro_batches = self.num_micro_batches
        grad_avals = self.grad_avals
        num_grads = len(grad_avals)
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host
        num_outs = len(self.out_avals)

        # Split batch argument into micro batches
        args = list(args)
        next_batches = []
        for i in self.global_batch_arg_indices:
            arg = args[i]
            new_shape = (num_micro_batches,
                         arg.shape[0] // num_micro_batches) + arg.shape[1:]
            reshaped = arg.reshape(new_shape)
            micro_batches = jnp.split(reshaped, num_micro_batches)
            micro_batches = [x.squeeze(0) for x in micro_batches]

            # Put the first micro batch in args.
            # Put the following micro batches to next_batches.
            args[i] = micro_batches[0]
            next_batches.extend(micro_batches[1:])

        # Shard arguments
        input_bufs = physical_mesh.shard_args(self.global_arg_shard_indices,
                                              self.donated_invars, args)
        next_batch_bufs = physical_mesh.shard_args(
            self.next_batch_indices, (True,) * len(self.next_batch_indices),
            next_batches)

        if physical_mesh.is_distributed:
            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_uuids = get_uuid_np_array(input_bufs)\
                .reshape(len(input_bufs), num_hosts, num_devices_per_host)\
                .transpose([1, 0, 2])

            if next_batch_bufs:
                next_batch_uuids = get_uuid_np_array(next_batch_bufs)\
                    .reshape(len(next_batch_bufs), num_hosts, num_devices_per_host)\
                    .transpose([1, 0, 2])
            else:
                next_batch_uuids = (None,) * num_hosts

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = next_remote_buffer_uuid(num_hosts * num_outs * num_devices_per_host)\
                .reshape(num_hosts, num_outs, num_devices_per_host)

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, input_uuids[i], next_batch_uuids[i],
                    output_uuids[i])

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, total_devices)
            output_bufs = np.empty((num_outs, physical_mesh.total_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh,
                        host_id,
                        device_id,
                        output_uuids[i][host_id][device_id],
                        dtype=self.out_avals[i].dtype)

            # Mark donated input buffers as already deleted on workers.
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.set_deleted_on_workers()

            # Mark micro batch buffers as already deleted on workers.
            for bufs in next_batch_bufs:
                for buf in bufs:
                    buf.set_deleted_on_workers()
        else:
            # Prepare gradient buffers
            timers(self.timer_name).start(self.sync_func)
            grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
                [])

            # Call accumulate_grad multiple times
            tmp_input_bufs = [input_bufs[i] for i in self.accumulate_grad_invar_indices] +\
                            grad_bufs
            os.environ[
                self.skip_allreduce_env_name] = self.grad_sync_channel_ids
            for i in range(num_micro_batches):
                if i != 0:
                    # Feed in the data of the next batch
                    tmp_input_bufs[-num_grads:] = grad_bufs
                    for j, idx in enumerate(
                            self.accumulate_grad_batch_arg_indices):
                        tmp_input_bufs[idx] = next_batch_bufs[
                            j * (num_micro_batches - 1) + (i - 1)]
                if i == num_micro_batches - 1:
                    os.environ[self.skip_allreduce_env_name] = ""
                grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                    tmp_input_bufs)

            # Call apply_grad
            tmp_input_bufs = [input_bufs[i] for i in self.apply_grad_invar_indices] +\
                            grad_bufs
            output_bufs = self.apply_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)
            timers(self.timer_name).stop(self.sync_func)

        # Wrap output buffers as ShardedArray
        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        raise NotImplementedError

    def profile_with_dummy_inputs(self, **kwargs):
        """Profile the time cost of this executable with dummy inputs."""
        raise NotImplementedError

    def get_execution_time_costs(self, warmup):
        """Return the pure execution time costs recorded by an internal timer."""
        return self.physical_mesh.get_remote_timer(
            self.timer_name).costs[warmup:]

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        if self.physical_mesh.is_distributed:
            return ray.get(self.physical_mesh.workers[0].\
                get_exec_total_allocation_size.remote(self.exec_uuid))
        else:
            return max(self.accumulate_grad.total_allocation_size(),
                       self.apply_grad.total_allocation_size())

    def get_hlo_text(self):
        return self.hlo_text

    def __del__(self):
        self.physical_mesh.delete_remote_executable(self)


class GradAccMeshWorkerExecutable:
    """The worker part of a gradient accumulation mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 accumulate_grad_proto: bytes, apply_grad_proto: bytes,
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 accumulate_grad_batch_arg_indices: Sequence[int],
                 grad_shard_shapes: Sequence[Tuple[int, ...]],
                 grad_shard_dtypes: Sequence[jnp.dtype],
                 strategy_config: StrategyConfig,
                 donated_invars: Sequence[bool], batch_invars: Sequence[bool],
                 num_grads: int, num_micro_batches: int,
                 grad_sync_channel_ids: str):
        from parax.shard_parallel.auto_sharding import (
            compile_with_given_strategy, HloProtoStatus)

        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())
        hlo_proto_status = HloProtoStatus.FULLY_OPTIMIZED

        self.accumulate_grad = compile_with_given_strategy(
            worker.backend, xla_client.XlaComputation(accumulate_grad_proto),
            strategy_config, num_devices, False, hlo_proto_status)
        self.apply_grad = compile_with_given_strategy(
            worker.backend, xla_client.XlaComputation(apply_grad_proto),
            strategy_config, num_devices, False, hlo_proto_status)
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
        self.grad_sync_channel_ids = grad_sync_channel_ids
        self.skip_allreduce_env_name =\
            self.accumulate_grad.hlo_modules()[0].name() +\
            "XLA_SKIP_NCCL_COLLECTIVE_IDS"

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self, input_uuids, next_batch_uuids, output_uuids):
        """Run the executable on the worker."""
        buffer_dict = self.buffer_dict
        num_micro_batches = self.num_micro_batches

        tmp_input_bufs = [
            get_buffers(buffer_dict, input_uuids[i])
            for i in self.accumulate_grad_invar_indices
        ]

        # Prepare gradient buffers
        timers(self.timer_name).start(self.sync_func)
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
                        next_batch_uuids[j * (num_micro_batches - 1) + (i - 1)])
            if i == num_micro_batches - 1:
                os.environ[self.skip_allreduce_env_name] = ""
            grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)

        # Call apply_grad
        tmp_input_bufs = [
            get_buffers(buffer_dict, input_uuids[i])
            for i in self.apply_grad_invar_indices
        ] + grad_bufs
        output_bufs = self.apply_grad.execute_sharded_on_local_devices(
            tmp_input_bufs)
        timers(self.timer_name).stop(self.sync_func)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, input_uuids)

        # Delete micro batch buffers
        for i in range(len(next_batch_uuids)):
            for j in range(len(next_batch_uuids[i])):
                del buffer_dict[next_batch_uuids[i][j]]

    def profile_with_dummy_inputs(self, backend, local_devices, **kwargs):
        """Profile the time cost of this executable with dummy inputs."""
        raise NotImplementedError

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        return max(self.accumulate_grad.total_allocation_size(),
                   self.apply_grad.total_allocation_size())

    def __del__(self):
        self.accumulate_grad.delete()
        self.apply_grad.delete()
        self.allocate_zero_buffers.delete()


def get_grad_sync_channel_ids_with_hint(hlo_module, hint) -> str:
    return xla_extension.get_grad_sync_channel_ids(hlo_module, hint)


class PartialGradAccMeshDriverExecutable(NormalMeshDriverExecutable):
    """
    The driver part of a mesh executable that 
    only compute and accumulate grads, but not apply it.
    """

    def __init__(self, physical_mesh: "PhysicalDeviceMesh", compiled,
                 strategy_config: StrategyConfig, avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 out_acc_grad_indices: Sequence[int]):
        if isinstance(compiled, ProtoAndSharding):
            hlo_module = hlo_module = xla_client.XlaComputation(
                compiled.proto).as_hlo_module()
        else:
            hlo_module = compiled.hlo_modules()[0]
        if physical_mesh.total_devices > 1:
            self.grad_sync_channel_ids = get_grad_sync_channel_ids_with_hint(
                hlo_module, out_acc_grad_indices)
        else:
            self.grad_sync_channel_ids = ""
        self.skip_allreduce_env_name =\
            hlo_module.name() + "XLA_SKIP_NCCL_COLLECTIVE_IDS"
        super(PartialGradAccMeshDriverExecutable,
              self).__init__(physical_mesh, compiled, strategy_config, avals,
                             out_avals, donated_invars)

    def set_executable(self, physical_mesh, compiled, strategy_config):
        hlo_module = self.hlo_module
        if physical_mesh.is_distributed:
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        PartialGradAccMeshWorkerExecutable,
                                        hlo_proto, strategy_config,
                                        self.grad_sync_channel_ids)
        else:
            self.compiled = compiled
            self.grad_sync_channel_ids = self.grad_sync_channel_ids

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        assert 'skip_grad_sync' in kwargs,\
            'Partial grad acc mesh executable missing kwargs "skip_grad_sync"'
        skip_grad_sync = kwargs['skip_grad_sync']
        if not self.physical_mesh.is_distributed:
            os.environ[self.skip_allreduce_env_name] =\
                self.grad_sync_channel_ids if skip_grad_sync else ""
        return super(PartialGradAccMeshDriverExecutable,
                     self).launch_on_driver(*args, **kwargs)


class PartialGradAccMeshWorkerExecutable(NormalMeshWorkerExecutable):
    """
    The worker part of a mesh executable that
    only compute and accumulate grads, but not apply it
    """

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo_proto: bytes,
                 strategy_config: StrategyConfig, grad_sync_channel_ids: str):
        super(PartialGradAccMeshWorkerExecutable,
              self).__init__(worker, uuid, hlo_proto, strategy_config)
        self.grad_sync_channel_ids = grad_sync_channel_ids
        self.skip_allreduce_env_name =\
            self.compiled.hlo_modules()[0].name() + "XLA_SKIP_NCCL_COLLECTIVE_IDS"

    def execute_on_worker(self,
                          input_uuids: List[List[int]],
                          output_uuids: List[List[int]],
                          skip_grad_sync=False,
                          **kwargs):
        """Run the executable on the worker."""
        os.environ[self.skip_allreduce_env_name] =\
            self.grad_sync_channel_ids if skip_grad_sync else ""
        return super(PartialGradAccMeshWorkerExecutable,
                     self).execute_on_worker(input_uuids, output_uuids,
                                             **kwargs)


class AllocZeroBufferDriverExecutable:
    """The driver part of a buffer-allocation executable."""

    def __init__(self, physical_mesh, grad_vars, grad_sharding_specs):
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
        if physical_mesh.is_distributed:
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        AllocZeroBufferWorkerExecutable,
                                        grad_shard_shapes, grad_shard_dtypes)
        else:
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                xla_bridge.get_backend("gpu"), physical_mesh.devices,
                grad_shard_shapes, grad_shard_dtypes)

        self.timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def get_driver_callable(self):
        ret = partial(self.launch_on_driver)
        ret.shard_args_only = partial(self.shard_args_only)
        ret.get_executable = lambda: self
        return ret

    def launch_on_driver(self, *args):
        assert len(args) == 0,\
            f"allocate zero buffers does not need args, got {len(args)}"
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_outs = len(self.out_avals)
        num_devices_per_host = physical_mesh.num_devices_per_host

        if physical_mesh.is_distributed:
            # Get output uuids
            output_uuids = next_remote_buffer_uuid(num_hosts * num_outs * num_devices_per_host)\
                .reshape(num_hosts, num_outs, num_devices_per_host)

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, [], output_uuids[i])

            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather outputs
            output_bufs = np.empty((num_outs, physical_mesh.total_devices),
                                   dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh,
                        host_id,
                        device_id,
                        output_uuids[i][host_id][device_id],
                        dtype=self.out_avals[i].dtype)
        else:
            timers(self.timer_name).start(self.sync_func)
            output_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
                [])
            timers(self.timer_name).stop(self.sync_func)

        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        raise NotImplementedError

    def profile_with_dummy_inputs(self):
        raise NotImplementedError

    def get_total_allocation_size(self):
        return 0

    def get_hlo_text(self):
        return ""

    def __del__(self):
        self.physical_mesh.delete_remote_executable(self)


class AllocZeroBufferWorkerExecutable:
    """The worker part of a buffer-allocation executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 grad_shard_shapes: Sequence[Tuple[int, ...]],
                 grad_shard_dtypes: Sequence[jnp.dtype]):
        num_devices = len(worker.backend.devices())
        self.allocate_zero_buffers = compile_allocate_zero_buffers(
            worker.backend, num_devices, grad_shard_shapes, grad_shard_dtypes)
        self.worker = worker

        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids,
                          output_uuids,
                          sync_before=False,
                          sync_after=False):
        buffer_dict = self.worker.buffers
        before_sync_func = self.sync_func if sync_before else None
        after_sync_func = self.sync_func if sync_after else None

        # Execute
        timers(self.timer_name).start(before_sync_func)
        grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices(
            [])
        timers(self.timer_name).stop(after_sync_func)
        for i in range(len(output_uuids)):
            set_buffers(buffer_dict, output_uuids[i], grad_bufs[i])

    def get_total_allocation_size(self):
        return 0

    def __del__(self):
        self.allocate_zero_buffers.delete()


class MemzeroWorkerExecutable:

    def __init__(self, worker: "MeshHostWorker", uuid: int, buffer_shard_shapes,
                 buffer_shard_dtypes):
        num_devices = len(worker.local_devices)
        self.memzero = compile_memset_zero_buffers(worker.backend, num_devices,
                                                   buffer_shard_shapes,
                                                   buffer_shard_dtypes)
        self.worker = worker

        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self,
                          input_uuids,
                          output_uuids,
                          sync_before=False,
                          sync_after=False):
        buffer_dict = self.worker.buffers
        before_sync_func = self.sync_func if sync_before else None
        after_sync_func = self.sync_func if sync_after else None

        # Get input
        input_bufs = [get_buffers(buffer_dict, x) for x in input_uuids]
        # Execute
        timers(self.timer_name).start(before_sync_func)
        _ = self.memzero.execute_sharded_on_local_devices(input_bufs)
        timers(self.timer_name).stop(after_sync_func)
