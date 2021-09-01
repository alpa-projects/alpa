from typing import Union, List

from jax._src.util import partial
from jax.interpreters import pxla
from jax.lib import xla_client, xla_bridge
import jax.numpy as jnp
import numpy as np
import ray

from parax.device_mesh import RemoteBufferRef
from parax.util import profile_xla_executable

remote_executable_counter = 0


def next_remote_executable_uuid():
    """Return the next uuid of a remote executable."""
    global remote_executable_counter
    remote_executable_counter = (remote_executable_counter + 1) % (1 << 60)
    return remote_executable_counter


def get_uuid_np_array(array):
    """Convert a np array of RemoteBufferRef to a np array of UUID (int64)."""
    ret = np.empty(array.shape, dtype=np.int64)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            ret[i][j] = array[i][j].uuid
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

        # Make arguments and outputs handler
        input_indices = [pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, self.input_sharding_specs)]
        self.args_handler = partial(physical_mesh.shard_args, input_indices, donated_invars)
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

        input_bufs = self.args_handler(args)
        if not physical_mesh.is_distributed:
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
        else:
            # TODO(lmzheng): reduce the overhead of meta information maintainance
            # by overlapping GPU computation and python code.

            # Donate input buffers
            for bufs, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    for buf in bufs:
                        buf.donate()

            # Shape: (num_hosts, num_args, num_devices_per_host)
            input_bufs = np.array(input_bufs) \
                .reshape(len(args), num_hosts, num_devices_per_host) \
                .transpose([1, 0, 2])

            # Allocate output buffer references
            # Shape: (num_hosts, num_outs, num_devices_per_host)
            output_bufs = np.empty((num_hosts, num_outs, num_devices_per_host), dtype=object)
            for i in range(physical_mesh.num_hosts):
                for j in range(num_outs):
                    for k in range(num_devices_per_host):
                        output_bufs[i][j][k] = RemoteBufferRef(physical_mesh, i, k)

            # Execute SPMD binary
            for i in range(num_hosts):
                host_inputs = get_uuid_np_array(input_bufs[i])
                host_outputs = get_uuid_np_array(output_bufs[i])
                physical_mesh.workers[i].run_executable.remote(self.remote_uuid, host_inputs,
                                                               host_outputs)

            # Gather outputs
            # Shape: (num_outs, total_devices)
            output_bufs = output_bufs.transpose([1, 0, 2]).reshape(
                (num_outs, physical_mesh.total_devices))
        return self.outs_handler(output_bufs)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        input_bufs = self.args_handler(args)
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


class NormalMeshWorkerExecutable:
    """The worker part of a normal mesh executable."""
    def __init__(self, backend, hlo_proto: bytes, strategy_config: "StrategyConfig"):
        from parax.shard_parallel.auto_sharding import (
            compile_with_given_strategy, HloProtoStatus)

        xla_computation = xla_client.XlaComputation(hlo_proto)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        assert num_devices == len(backend.devices())
        hlo_proto_status = HloProtoStatus.FULLY_OPTIMIZED

        self.compiled = compile_with_given_strategy(backend, xla_computation,
                                                    strategy_config, num_devices,
                                                    False, hlo_proto_status)

    def execute_on_worker(self, local_buffers, input_uuids: List[List[int]],
                output_uuids: List[List[int]]):
        """Run the executable on the worker."""
        # Map uuids to input buffers
        device_inputs = [[None
                          for _ in range(input_uuids.shape[1])]
                         for _ in range(input_uuids.shape[0])]
        for i in range(input_uuids.shape[0]):
            for j in range(input_uuids.shape[1]):
                device_inputs[i][j] = local_buffers[input_uuids[i][j]]

        # Execute the executable
        device_outs = self.compiled.execute_sharded_on_local_devices(device_inputs)

        # Store output buffers
        for i in range(output_uuids.shape[0]):
            for j in range(output_uuids.shape[1]):
                local_buffers[output_uuids[i][j]] = device_outs[i][j]

        # Delete donated input buffers
        for i in range(input_uuids.shape[0]):
            for j in range(input_uuids.shape[1]):
                if device_inputs[i][j].is_deleted():
                    del local_buffers[input_uuids[i][j]]

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

        input_sharding_specs = [None] * len(avals)
        for i, idx in enumerate(accumulate_grad_invar_indices):
            input_sharding_specs[idx] = accumulate_grad_input_sharding_specs[i]
        for i, idx in enumerate(apply_grad_invar_indices):
            if input_sharding_specs[idx] is None:
                input_sharding_specs[idx] = apply_grad_input_sharding_specs[i]
            else:
                assert input_sharding_specs[idx] == apply_grad_input_sharding_specs[
                    i]
        num_grads = len(grad_avals)
        assert accumulate_grad_input_sharding_specs[-num_grads:] == grad_sharding_specs

        self.total_allocation_size = max(accumulate_grad.total_allocation_size(),
                                         apply_grad.total_allocation_size())

        # Make arguments and outputs handler
        self.global_in_shard_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, input_sharding_specs)
        ]
        self.grad_shard_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        self.input_buffer_batch_arg_indices = []
        for input_idx, global_idx in enumerate(accumulate_grad_invar_indices):
            if batch_invars[global_idx]:
                self.input_buffer_batch_arg_indices.append((input_idx, global_idx))
        self.outs_handler = physical_mesh.get_outputs_handler(out_avals, output_sharding_specs)

        # Send the executable to workers
        if physical_mesh.is_distributed:
            assert False
        else:
            self.remote_uuid = None
            self.accumulate_grad = accumulate_grad
            self.apply_grad = apply_grad

    def get_driver_callable(self):
        """Get a callable that runs on the driver and handles arguments/outputs conversion."""
        ret = partial(self.launch_on_driver)
        ret.shard_args_only = partial(self.shard_args_only)
        ret.get_executable = lambda: self
        return ret

    def launch_on_driver(self, *global_args):
        """Launch the executable on the driver."""
        num_micro_batches = self.num_micro_batches
        grad_avals = self.grad_avals
        num_grads = len(grad_avals)
        devices = self.physical_mesh.devices
        global_in_shard_indices = self.global_in_shard_indices

        # Shard global input arguments
        global_buffers = []
        for i, arg in enumerate(global_args):
            if self.batch_invars[i]:
                new_shape = (num_micro_batches,
                             arg.shape[0] // num_micro_batches) + arg.shape[1:]
                reshaped = arg.reshape(new_shape)
                micro_batches = jnp.split(reshaped, num_micro_batches)
                micro_batches = [x.squeeze(0) for x in micro_batches]
                micro_batches = pxla.shard_args(
                    devices, (global_in_shard_indices[i],) * len(micro_batches),
                    micro_batches)
                global_buffers.append(micro_batches)
            else:
                global_buffers.append(
                    pxla.shard_args(devices, [global_in_shard_indices[i]],
                                    [arg])[0])

        # Prepare gradient buffers
        grad_buffers = pxla.shard_args(devices, self.grad_shard_indices, [
            jnp.zeros(grad_avals[i].shape, grad_avals[i].dtype)
            for i in range(num_grads)
        ])

        # Call accumulate_grad multiple times
        input_buffers = [global_buffers[i] for i in self.accumulate_grad_invar_indices] +\
                        grad_buffers
        for j in range(num_micro_batches):
            for input_idx, global_idx in self.input_buffer_batch_arg_indices:
                input_buffers[input_idx] = global_buffers[global_idx][j]
            input_buffers[-num_grads:] = grad_buffers
            grad_buffers = self.accumulate_grad.execute_sharded_on_local_devices(
                input_buffers)

        # Call apply_grad
        input_buffers = [global_buffers[i] for i in self.apply_grad_invar_indices] +\
                        grad_buffers
        output_buffers = self.apply_grad.execute_sharded_on_local_devices(
            input_buffers)

        # Wrap global output buffers as ShardedArray
        return self.outs_handler(output_buffers)

    def shard_args_only(self, *args):
        """Pre-shard the input arguments."""
        raise NotImplementedError

    def profile_with_dummy_inputs(self):
        """Profile the time cost of this executable with dummy inputs."""
        raise NotImplementedError

    def get_total_allocation_size(self):
        """Get the total allocated memory size of this executable."""
        raise NotImplementedError
