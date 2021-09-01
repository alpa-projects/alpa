from typing import Union, List

from jax._src.util import partial
from jax.interpreters import pxla
from jax.lib import xla_client, xla_bridge
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
        physical_mesh = self.physical_mesh

        input_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(self.avals, self.input_sharding_specs)
        ]

        args_handler = partial(physical_mesh.shard_args, input_indices, self.donated_invars)
        outs_handler = physical_mesh.get_outputs_handler(self.out_avals, self.output_sharding_specs)

        ret = partial(self.execute_on_driver, args_handler, outs_handler,
                      len(self.out_avals), self.donated_invars)
        ret.shard_args_only = partial(self.shard_args_only, args_handler, self.avals,
                                      self.input_sharding_specs)
        ret.get_executable = lambda: self
        return ret

    def execute_on_driver(self, args_handler, outs_handler, num_outs, donated_invars, *args):
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_devices_per_host = physical_mesh.num_devices_per_host

        input_bufs = args_handler(args)
        if not physical_mesh.is_distributed:
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
        else:
            # TODO(lmzheng): reduce the overhead of meta information maintainance
            # by overlapping GPU computation and python code.

            # Donate input buffers
            for bufs, is_donated in zip(input_bufs, donated_invars):
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
        return outs_handler(output_bufs)

    def shard_args_only(self, args_handler, avals, sharding_specs, *args):
        """Pre-shard the input arguments."""
        input_bufs = args_handler(args)
        outs_handler = self.physical_mesh.get_outputs_handler(self.avals, self.input_sharding_specs)
        return outs_handler(input_bufs)

    def profile_with_dummy_inputs(self):
        """Profile the time cost of an xla executable."""
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
        return profile_xla_executable(self.compiled,
                                      backend, local_devices)


class GradAccMeshExecutable:
    def __init__(self):
        pass

