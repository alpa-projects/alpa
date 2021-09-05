from typing import List, Sequence, Tuple, Union

import jax
from jax._src.util import partial
from jax.interpreters import pxla
from jax.lib import xla_client, xla_bridge
import jax.numpy as jnp
import numpy as np
import ray

from parax.device_mesh import RemoteBufferRef, next_remote_buffer_uuid
from parax.util import compile_allocate_zero_buffers, get_shard_shape, profile_xla_executable

remote_executable_counter = 0


def next_remote_executable_uuid():
    """Return the next uuid of a remote executable."""
    global remote_executable_counter
    remote_executable_counter = (remote_executable_counter + 1) % (1 << 60)
    return remote_executable_counter


def get_uuid_np_array(array):
    """Convert a 2d array of RemoteBufferRef to a np array of UUID (int64)."""
    shape = (len(array), len(array[0]))
    ret = np.empty(shape, dtype=np.int64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            ret[i,j] = array[i][j].uuid
    return ret


class NormalMeshDriverExecutable:
    """The driver part of a normal mesh executable."""
    def __init__(self, physical_mesh, compiled, strategy_config,
                 avals, out_avals, donated_invars):
        from parax.shard_parallel.auto_sharding import get_input_output_sharding_specs

        self.physical_mesh = physical_mesh
        self.avals = avals
        self.out_avals = out_avals
        self.donated_invars = donated_invars

        # Read sharding specs
        hlo_module = compiled.hlo_modules()[0]
        self.input_sharding_specs, self.output_sharding_specs = get_input_output_sharding_specs(
            hlo_module, physical_mesh.total_devices, avals,
            out_avals, strategy_config.logical_mesh_shape)
        self.total_allocation_size = compiled.total_allocation_size()

        # Cache results for input and output sharding
        self.input_indices = [pxla.spec_to_indices(aval.shape, spec)
                              for aval, spec in zip(avals, self.input_sharding_specs)]
        self.outs_handler = physical_mesh.get_outputs_handler(out_avals, self.output_sharding_specs)

        # Send the executable to workers
        if physical_mesh.is_distributed:
            self.remote_uuid = next_remote_executable_uuid()
            hlo_proto = hlo_module.as_serialized_hlo_module_proto()
            for w in physical_mesh.workers:
                w.put_executable.remote(self.remote_uuid, NormalMeshWorkerExecutable,
                                        hlo_proto, strategy_config)
        else:
            self.remote_uuid = None
            self.compiled = compiled

    def get_driver_callable(self):
        """Get a callable that runs on the driver and handles arguments/outputs conversion."""
        ret = partial(self.launch_on_driver)
        ret.shard_args_only = partial(self.shard_args_only)
        ret.get_executable = lambda: self
        return ret

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host
        num_outs = len(self.out_avals)

        input_bufs = physical_mesh.shard_args(self.input_indices, self.donated_invars, args)
        if physical_mesh.is_distributed:
            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_uuids = get_uuid_np_array(input_bufs)\
                .reshape(len(args), num_hosts, num_devices_per_host)\
                .transpose([1, 0, 2])

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = next_remote_buffer_uuid(num_hosts * num_outs * num_devices_per_host)\
                .reshape(num_hosts, num_outs, num_devices_per_host)\

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.remote_uuid, input_uuids[i], output_uuids[i])

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, total_devices)
            output_bufs = np.empty((num_outs, physical_mesh.total_devices), dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id, output_uuids[i][host_id][device_id])

            # Donate input buffers
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.donate()
        else:
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)

        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        input_bufs = self.physical_mesh.shard_args(self.input_indices, self.donated_invars, args)
        outs_handler = self.physical_mesh.get_outputs_handler(self.avals, self.input_sharding_specs)
        return outs_handler(input_bufs)

    def profile_with_dummy_inputs(self):
        """Profile the time cost of this executable with dummy inputs."""
        if self.physical_mesh.is_distributed:
            tasks = []
            for worker in self.physical_mesh.workers:
                tasks.append(worker.profile_executable_with_dummy_inputs.remote(self.remote_uuid))
            costs = ray.get(tasks)[0]
        else:
            costs = profile_xla_executable(self.compiled,
                                           xla_bridge.get_backend("gpu"),
                                           self.physical_mesh.devices)
        return costs

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        return self.total_allocation_size


def get_buffers(local_buffers, uuids):
    """Get buffers by uuids from the local buffer dictionary."""
    return [local_buffers[uuid] for uuid in uuids]


def set_buffers(local_buffers, uuids, buffers):
    """Store buffers to the local buffer dictionary."""
    for uuid, buf in zip(uuids, buffers):
        local_buffers[uuid] = buf


def delete_donated_buffers(local_buffers, uuids):
    """Delete the donated buffers from the local buffer dictionary."""
    for i in range(len(uuids)):
        for j in range(len(uuids[i])):
            uuid = uuids[i][j]
            if isinstance(uuid, (np.int64, int)) and local_buffers[uuid].is_deleted():
                del local_buffers[uuid]


class NormalMeshWorkerExecutable:
    """The worker part of a normal mesh executable."""
    def __init__(self, worker, hlo_proto: bytes, strategy_config: "StrategyConfig"):
        from parax.shard_parallel.auto_sharding import (
            compile_with_given_strategy, HloProtoStatus)

        xla_computation = xla_client.XlaComputation(hlo_proto)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())
        hlo_proto_status = HloProtoStatus.FULLY_OPTIMIZED

        self.compiled = compile_with_given_strategy(worker.backend, xla_computation,
                                                    strategy_config, num_devices,
                                                    False, hlo_proto_status)
        self.local_buffers = worker.local_buffers

    def execute_on_worker(self, input_uuids: List[List[int]], output_uuids: List[List[int]]):
        """Run the executable on the worker."""
        local_buffers = self.local_buffers

        # Get input buffers from uuids
        input_bufs = [get_buffers(local_buffers, x) for x in input_uuids]

        # Execute the executable
        output_bufs = self.compiled.execute_sharded_on_local_devices(input_bufs)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(local_buffers, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(local_buffers, input_uuids)

    def profile_with_dummy_inputs(self, backend, local_devices):
        """Profile the time cost of this executable with dummy inputs."""
        return profile_xla_executable(self.compiled,
                                      backend, local_devices)


class GradAccMeshDriverExecutable:
    def __init__(self, physical_mesh, accumulate_grad, apply_grad, strategy_config,
                 avals, out_avals, grad_avals, donated_invars, batch_invars,
                 accumulate_grad_invar_indices, apply_grad_invar_indices,
                 num_micro_batches):
        from parax.shard_parallel.auto_sharding import get_input_output_sharding_specs

        self.physical_mesh = physical_mesh
        self.avals = avals
        self.out_avals = out_avals
        self.grad_avals = grad_avals
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.num_micro_batches = num_micro_batches

        # Read sharding specs
        accumulate_grad_in_avals = [avals[i] for i in accumulate_grad_invar_indices] + grad_avals
        apply_grad_in_avals = [avals[i] for i in apply_grad_invar_indices] + grad_avals
        accumulate_grad_input_sharding_specs, grad_sharding_specs =\
            get_input_output_sharding_specs(
            accumulate_grad.hlo_modules()[0], physical_mesh.total_devices,
            accumulate_grad_in_avals, grad_avals, strategy_config.logical_mesh_shape)
        apply_grad_input_sharding_specs, output_sharding_specs =\
            get_input_output_sharding_specs(
            apply_grad.hlo_modules()[0], physical_mesh.total_devices,
            apply_grad_in_avals, out_avals, strategy_config.logical_mesh_shape)

        global_arg_sharding_specs = [None] * len(avals)
        for i, idx in enumerate(accumulate_grad_invar_indices):
            global_arg_sharding_specs[idx] = accumulate_grad_input_sharding_specs[i]
        for i, idx in enumerate(apply_grad_invar_indices):
            if global_arg_sharding_specs[idx] is None:
                global_arg_sharding_specs[idx] = apply_grad_input_sharding_specs[i]
            else:
                assert global_arg_sharding_specs[idx] == apply_grad_input_sharding_specs[
                    i]
        num_grads = len(grad_avals)
        assert accumulate_grad_input_sharding_specs[-num_grads:] == grad_sharding_specs

        self.total_allocation_size = max(accumulate_grad.total_allocation_size(),
                                         apply_grad.total_allocation_size())

        # Cache results for input and output sharding
        global_arg_shard_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, global_arg_sharding_specs)
        ]
        global_batch_arg_indices = [i for i in range(len(avals)) if batch_invars[i]]
        accumulate_grad_batch_arg_indices = [i for i, j in enumerate(accumulate_grad_invar_indices) if batch_invars[j]]
        next_batch_indices = []
        for i in global_batch_arg_indices:
            next_batch_indices.extend([global_arg_shard_indices[i]] * (num_micro_batches - 1))
        grad_shard_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        grad_shard_dtypes = [aval.dtype for aval in grad_avals]
        self.outs_handler = physical_mesh.get_outputs_handler(out_avals, output_sharding_specs)
        self.global_batch_arg_indices = global_batch_arg_indices
        self.next_batch_indices = next_batch_indices
        self.global_arg_shard_indices = global_arg_shard_indices

        # Send the executable to workers
        if physical_mesh.is_distributed:
            self.remote_uuid = next_remote_executable_uuid()
            for w in physical_mesh.workers:
                w.put_executable.remote(
                    self.remote_uuid,
                    GradAccMeshWorkerExecutable,
                    accumulate_grad.hlo_modules()[0].as_serialized_hlo_module_proto(),
                    apply_grad.hlo_modules()[0].as_serialized_hlo_module_proto(),
                    accumulate_grad_invar_indices,
                    apply_grad_invar_indices,
                    accumulate_grad_batch_arg_indices,
                    grad_shard_shapes,
                    grad_shard_dtypes,
                    strategy_config,
                    donated_invars,
                    batch_invars,
                    num_grads,
                    num_micro_batches)
        else:
            self.remote_uuid = None
            self.accumulate_grad = accumulate_grad
            self.apply_grad = apply_grad
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                xla_bridge.get_backend("gpu"), physical_mesh.total_devices,
                grad_shard_shapes, grad_shard_dtypes)
            self.accumulate_grad_batch_arg_indices = accumulate_grad_batch_arg_indices

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
        donated_invars = self.donated_invars
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
        input_bufs = physical_mesh.shard_args(self.global_arg_shard_indices, self.donated_invars, args)
        next_batch_bufs = physical_mesh.shard_args(self.next_batch_indices, 
            (False,) * len(self.next_batch_indices), next_batches)

        if physical_mesh.is_distributed:
            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_uuids = get_uuid_np_array(input_bufs)\
                .reshape(len(input_bufs), num_hosts, num_devices_per_host)\
                .transpose([1, 0, 2])

            next_batch_uuids = get_uuid_np_array(next_batch_bufs)\
                .reshape(len(next_batch_bufs), num_hosts, num_devices_per_host)\
                .transpose([1, 0, 2])

            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_uuids = next_remote_buffer_uuid(num_hosts * num_outs * num_devices_per_host)\
                .reshape(num_hosts, num_outs, num_devices_per_host)\

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.remote_uuid, input_uuids[i], next_batch_uuids[i], output_uuids[i])

            # Shape: (num_outs, num_hosts, num_devices_per_host)
            output_uuids = output_uuids.transpose([1, 0, 2])

            # Gather output buffers
            # Shape: (num_outs, total_devices)
            output_bufs = np.empty((num_outs, physical_mesh.total_devices), dtype=object)
            for i in range(len(output_bufs)):
                for j in range(len(output_bufs[i])):
                    host_id = j // num_devices_per_host
                    device_id = j % num_devices_per_host
                    output_bufs[i][j] = RemoteBufferRef(
                        physical_mesh, host_id, device_id, output_uuids[i][host_id][device_id])

            # Donate input buffers
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.donate()
        else:
            # Prepare gradient buffers
            grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices([])

            # Call accumulate_grad multiple times
            tmp_input_bufs = [input_bufs[i] for i in self.accumulate_grad_invar_indices] +\
                            grad_bufs
            for i in range(num_micro_batches):
                if i != 0:
                    # Feed in the data of the next batch
                    tmp_input_bufs[-num_grads:] = grad_bufs
                    for j, idx in enumerate(self.accumulate_grad_batch_arg_indices):
                        tmp_input_bufs[idx] = next_batch_bufs[j * (num_micro_batches - 1) + (i-1)]
                grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                    tmp_input_bufs)

            # Call apply_grad
            tmp_input_bufs = [input_bufs[i] for i in self.apply_grad_invar_indices] +\
                            grad_bufs
            output_bufs = self.apply_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)

        # Wrap output buffers as ShardedArray
        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        raise NotImplementedError

    def profile_with_dummy_inputs(self):
        """Profile the time cost of this executable with dummy inputs."""
        raise NotImplementedError

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        raise NotImplementedError


class GradAccMeshWorkerExecutable:
    def __init__(self, worker,
                 accumulate_grad_proto: bytes,
                 apply_grad_proto: bytes,
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 accumulate_grad_batch_arg_indices: Sequence[int],
                 grad_shard_shapes: Sequence[Tuple[int, ...]],
                 grad_shard_dtypes: Sequence[jnp.dtype],
                 strategy_config: "StrategyConfig",
                 donated_invars: Sequence[bool],
                 batch_invars: Sequence[bool],
                 num_grads: int,
                 num_micro_batches: int):
        from parax.shard_parallel.auto_sharding import (
            compile_with_given_strategy, HloProtoStatus)

        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())
        hlo_proto_status = HloProtoStatus.FULLY_OPTIMIZED

        self.worker = worker
        self.accumulate_grad = compile_with_given_strategy(
            worker.backend, xla_client.XlaComputation(accumulate_grad_proto),
            strategy_config, num_devices, False, hlo_proto_status)
        self.apply_grad = compile_with_given_strategy(
            worker.backend, xla_client.XlaComputation(apply_grad_proto),
            strategy_config, num_devices, False, hlo_proto_status)
        self.allocate_zero_buffers = compile_allocate_zero_buffers(
            worker.backend, num_devices,
            grad_shard_shapes, grad_shard_dtypes)
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.accumulate_grad_batch_arg_indices = accumulate_grad_batch_arg_indices
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.num_grads = num_grads
        self.num_micro_batches = num_micro_batches

    def execute_on_worker(self, input_uuids, next_batch_uuids, output_uuids):
        """Run the executable on the worker."""
        local_buffers = self.worker.local_buffers
        num_micro_batches = self.num_micro_batches

        # Prepare gradient buffers
        grad_bufs = self.allocate_zero_buffers.execute_sharded_on_local_devices([])

        # Call accumulate_grad multiple times
        tmp_input_bufs = [get_buffers(local_buffers, input_uuids[i])
                             for i in self.accumulate_grad_invar_indices] + grad_bufs
        for i in range(num_micro_batches):
            if i != 0:
                # Feed in the data of the next batch
                tmp_input_bufs[-self.num_grads:] = grad_bufs
                for j, idx in enumerate(self.accumulate_grad_batch_arg_indices):
                    tmp_input_bufs[idx] = get_buffers(local_buffers,
                                 next_batch_uuids[j * (num_micro_batches - 1) + (i-1)])
            grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)

        # Call apply_grad
        tmp_input_bufs = [get_buffers(local_buffers, input_uuids[i])
                         for i in self.apply_grad_invar_indices] + grad_bufs
        output_bufs = self.apply_grad.execute_sharded_on_local_devices(
            tmp_input_bufs)

        # Store output buffers
        for i in range(len(output_uuids)):
            set_buffers(local_buffers, output_uuids[i], output_bufs[i])

        # Delete donated input buffers
        delete_donated_buffers(local_buffers, input_uuids)

    def profile_with_dummy_inputs(self, backend, local_devices):
        """Profile the time cost of this executable with dummy inputs."""
        raise NotImplementedError
