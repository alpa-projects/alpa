# pylint: disable=arguments-differ
"""A mesh executable encapsulates all compiled binary and meta information of
a distributed executable.

A mesh executable contains one or several XLA executables.
For each type of mesh executable, there is a driver part and a worker part.
The driver part runs on the user script and the worker parts run on distributed
workers. The driver part sends control commands to launch the worker parts on
workers.
"""
from abc import ABC, abstractmethod
from typing import Sequence, Optional
import os

from jax import xla
import jax.numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jax._src.lib import xla_client as xc, xla_extension as xe
from jax.core import ShapedArray
from jax.interpreters import pxla
from jax.tree_util import tree_flatten, tree_unflatten, tree_leaves, PyTreeDef
import numpy as np
import ray
from alpa.util import XlaPassContext

from alpa.device_mesh import (LocalPhysicalDeviceMesh,
                              DistributedPhysicalDeviceMesh, RemoteArrayRef,
                              next_array_uuids)
from alpa.global_env import global_config
from alpa.parallel_plan import (PlacementSpec, StagePlan, ClusterInfo,
                                ParallelPlan)
from alpa.shard_parallel.auto_sharding import (AutoShardingOption,
                                               get_input_output_sharding_specs,
                                               make_replicated_spec,
                                               run_backend_compilation,
                                               run_spmd_partitioner_pass)
from alpa.timer import timers
from alpa.util import (compile_allocate_zero_buffers, get_compile_options,
                       get_index_select_computation, get_shard_shape,
                       get_microbatch_sharding_spec, profile_xla_executable)
from alpa.wrapped_hlo import HloStatus, WrappedHlo


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

    def get_input_placement_specs(self):
        """
        Return the preferred placement specs for input arguments.
        The return value is a pytree of PlacementSpec
        with the same structure as the input pytree.
        """
        raise NotImplementedError()

    def get_output_placement_specs(self):
        """
        Return the preferred placement specs for outputs.
        The return value is a pytree of PlacementSpec
        with the same structure as the output pytree.
        """
        raise NotImplementedError()

    def get_parallel_plan(self):
        """Get the overall parallel plan."""
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

    def get_execution_time_costs(self):
        """Return the pure execution time costs recorded by an internal
        timer."""
        return self.physical_mesh.get_remote_timer(self.exec_timer_name).costs

    def get_shard_args_time_costs(self):
        """Return the time costs of sharding input arguments."""
        return timers(self.shard_args_timer_name).costs

    def get_hlo_text(self, status: HloStatus):
        """Return the HLO IR in the text format."""
        raise NotImplementedError()

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
        raise NotImplementedError()

    def dump_debug_info(self, folder: str):
        """
        Dump intermediate representations and other informations for debugging.
        """
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


# The global executable counter
mesh_executable_counter = 0


def next_mesh_executable_uuid():
    """Return the next uuid of a mesh executable."""
    global mesh_executable_counter
    mesh_executable_counter = (mesh_executable_counter + 1) % (1 << 60)
    return mesh_executable_counter


def get_execution_timer_name(exec_uuid: int):
    """Return the name of the timer used for recording pure execution time."""
    return f"exec-{exec_uuid}"


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


def wrap_to_placement_spec_tree(physical_mesh, avals, sharding_specs, pytree):
    """Wrap avals and sharding specs to a pytree of placement specs."""
    placement_specs = [
        PlacementSpec(aval, (physical_mesh.mesh_id,), (sharding_spec,))
        for aval, sharding_spec in zip(avals, sharding_specs)
    ]
    return tree_unflatten(pytree, placement_specs)


class NormalMeshDriverExecutable(MeshDriverExecutable):
    """The driver part of a normal mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 hlo: WrappedHlo,
                 stage_plan: StagePlan,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 static_argnums: Optional[Sequence[int]] = None,
                 in_tree: Optional[PyTreeDef] = None,
                 out_tree: Optional[PyTreeDef] = None,
                 flop_count: Optional[int] = None):
        self.physical_mesh = physical_mesh
        self.hlo = hlo
        self.avals = avals
        self.out_avals = out_avals
        self.donated_invars = donated_invars
        self.static_argnums = static_argnums
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.flop_count = flop_count
        self.stage_plan = stage_plan
        self.auto_sharding_option = stage_plan.auto_sharding_option
        self.auto_sharding_objective = stage_plan.auto_sharding_objective

        # Send the executable to workers
        self.fully_optimized_hlo_text = None
        self.exec_uuid = next_mesh_executable_uuid()
        self._set_executable(physical_mesh, hlo, stage_plan)

        if hlo.is_sharding_annotated():
            hlo = run_spmd_partitioner_pass(hlo, physical_mesh.num_devices)
        # Read sharding specs
        self.input_sharding_specs, self.output_sharding_specs = (
            get_input_output_sharding_specs(hlo.get_module(), avals, out_avals,
                                            physical_mesh.num_devices,
                                            stage_plan.logical_mesh_shape))

        # Cache results for input and output sharding
        self.input_indices = [
            pxla.spec_to_indices(aval.shape, spec)
            for aval, spec in zip(avals, self.input_sharding_specs)
        ]
        self.outs_handler = physical_mesh.get_outputs_handler(
            out_avals, self.output_sharding_specs)

        # Set up timers
        self.exec_timer_name = get_execution_timer_name(self.exec_uuid)
        self.shard_args_timer_name = self.exec_timer_name + "-shard-args"
        self.sync_func = get_sync_func_driver(physical_mesh)

    def _set_executable(self, physical_mesh, hlo, stage_plan):
        """Put the executable on workers."""
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        NormalMeshWorkerExecutable, hlo,
                                        stage_plan, self.donated_invars)
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)

            if physical_mesh.devices[0] is None:
                # A fake physical mesh for generating HLO module only
                self.compiled = run_backend_compilation(
                    physical_mesh.backend,
                    hlo,
                    stage_plan,
                    physical_mesh.num_devices,
                    bypass_device_assignment_check=True)
            else:
                self.compiled = run_backend_compilation(
                    physical_mesh.backend, hlo, stage_plan,
                    physical_mesh.num_devices)
            self.fully_optimized_hlo_text = self.compiled.hlo_modules(
            )[0].to_string()

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_outs = len(self.out_avals)

        timers(self.shard_args_timer_name).start()
        input_bufs = physical_mesh.shard_args_to_bufs(self.input_indices,
                                                      self.donated_invars,
                                                      (False,) * len(args),
                                                      None, args)
        timers(self.shard_args_timer_name).stop()

        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            input_uuids = np.array([ref.uuid for ref in input_bufs])
            output_uuids = next_array_uuids(num_outs)

            if "sync_before" not in kwargs:
                kwargs["sync_before"] = kwargs["sync_after"] = (
                    global_config.shard_parallel_sync_for_timer)

            # Execute the SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, input_uuids, output_uuids, **kwargs)

            # Gather output buffers
            output_bufs = np.array(
                [RemoteArrayRef(physical_mesh, uuid) for uuid in output_uuids])

            # Mark donated input buffers as already deleted on workers.
            for ary_ref, is_donated in zip(input_bufs, self.donated_invars):
                if is_donated:
                    ary_ref.set_deleted_on_workers()
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            sync_func = (self.sync_func if
                         global_config.shard_parallel_sync_for_timer else None)

            timers(self.exec_timer_name).start(sync_func)
            output_bufs = self.compiled.execute_sharded_on_local_devices(
                input_bufs)
            timers(self.exec_timer_name).stop(sync_func)

        return self.outs_handler(output_bufs)

    def get_input_placement_specs(self):
        """
        Return the preferred placement specs for input arguments.
        The return value is a pytree of PlacementSpec
        with the same structure as the input pytree.
        """
        return wrap_to_placement_spec_tree(self.physical_mesh, self.avals,
                                           self.input_sharding_specs,
                                           self.in_tree)

    def get_output_placement_specs(self):
        """
        Return the preferred placement specs for outputs.
        The return value is a pytree of PlacementSpec
        with the same structure as the output pytree.
        """
        return wrap_to_placement_spec_tree(self.physical_mesh, self.out_avals,
                                           self.output_sharding_specs,
                                           self.out_tree)

    def get_parallel_plan(self):
        """Get the overall parallel plan."""
        cluster_info = ClusterInfo(self.physical_mesh.num_hosts,
                                   self.physical_mesh.num_devices_per_host)
        return ParallelPlan(cluster_info, None, self.auto_sharding_option, None,
                            tree_leaves(self.get_input_placement_specs()))

    def preshard_dynamic_args(self, *args):
        """Pre-shard the input arguments."""
        input_bufs = self.physical_mesh.shard_args_to_bufs(
            self.input_indices, self.donated_invars, (False,) * len(args), None,
            args)
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
        return tree_unflatten(self.out_tree, out)

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
            costs = profile_xla_executable(self.compiled,
                                           self.physical_mesh.backend,
                                           self.physical_mesh.devices)
        return costs

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            return (ray.get(self.physical_mesh.workers[0].
                            get_exec_total_allocation_size.remote(
                                self.exec_uuid)))
        else:
            assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
            return self.compiled.total_allocation_size()

    def get_hlo_text(self, status: HloStatus = HloStatus.FULLY_OPTIMIZED):
        """Return the HLO IR in the text format."""
        if status == HloStatus.FULLY_OPTIMIZED:
            if self.fully_optimized_hlo_text is not None:
                return self.fully_optimized_hlo_text
            assert isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh)
            self.fully_optimized_hlo_text = ray.get(
                self.physical_mesh.workers[0].get_exec_hlo_text.remote(
                    self.exec_uuid))
            return self.fully_optimized_hlo_text
        else:
            raise ValueError(f"Invalid status: {status}")

    def dump_debug_info(self, folder: str):
        """
        Dump intermediate representations and other informations for debugging.
        """
        os.makedirs(folder, exist_ok=True)
        name = self.hlo.name
        name = name[:name.index("shard_parallel") - 1]
        prefix = os.path.join(folder, name)
        with open(f"{prefix}.hlo", "w") as f:
            f.write(self.get_hlo_text())
        with open(f"{prefix}.mem_usage.txt", "w") as f:
            f.write(f"total_allocation_size: "
                    f"{self.get_total_allocation_size()/(1024**3):.3f} GB\n")
        with open(f"{prefix}_input_placement_specs.txt", "w") as f:
            f.write(str(self.get_input_placement_specs()) + "\n\n")
        with open(f"{prefix}_output_placement_specs.txt", "w") as f:
            f.write(str(self.get_output_placement_specs()) + "\n\n")


def delete_donated_buffers(buffer_dict, uuids, donated_invars):
    """Delete the donated buffers from the local buffer dictionary."""
    for uuid, is_donated in zip(uuids, donated_invars):
        if is_donated:
            del buffer_dict[uuid]


class NormalMeshWorkerExecutable(MeshWorkerExecutable):
    """The worker part of a normal mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo: WrappedHlo,
                 stage_plan: StagePlan, donated_invars: Sequence[bool]):
        num_devices = np.prod(stage_plan.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())

        self.compiled = run_backend_compilation(worker.backend, hlo, stage_plan,
                                                num_devices)
        self.donated_invars = donated_invars
        self.worker = worker

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self, input_uuids: Sequence[int],
                          output_uuids: Sequence[int], sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input buffers from uuids
        # Sequence[Sequence[DeviceBuffer]], shape(num_args, num_devices)
        input_bufs = [buffer_dict[x] for x in input_uuids]

        if global_config.enable_overlapping:
            xe.computation_wait_events(input_uuids, self.worker.backend)
            xe.set_idx_to_uuid(output_uuids)
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
            buffer_dict[output_uuids[i]] = output_bufs[i]

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, input_uuids, self.donated_invars)

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
    """Return the channel ids of all-reduces that are used for gradient
    synchronization.

    The return value is a string containing all channel ids separated by
    periods. (e.g., ".0.12." means channel id 0 and 12)
    """
    return xe.get_grad_sync_channel_ids(hlo_module)


class GradAccMeshDriverExecutable(MeshDriverExecutable):
    """The driver part of a gradient accumulation mesh executable."""

    def __init__(self,
                 physical_mesh: "PhysicalDeviceMesh",
                 accumulate_grad: WrappedHlo,
                 apply_grad: WrappedHlo,
                 stage_plan: StagePlan,
                 avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 grad_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool],
                 batch_invars: Sequence[bool],
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 num_micro_batches: int,
                 in_tree: Optional[PyTreeDef] = None,
                 out_tree: Optional[PyTreeDef] = None,
                 flop_count: Optional[int] = None):
        self.physical_mesh = physical_mesh
        self.accumulate_grad_hlo = accumulate_grad
        self.apply_grad_hlo = apply_grad
        self.avals = avals
        self.out_avals = out_avals
        self.grad_avals = grad_avals
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.num_micro_batches = num_micro_batches
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.flop_count = flop_count
        self.stage_plan = stage_plan
        self.auto_sharding_option = stage_plan.auto_sharding_option
        self.auto_sharding_objective = stage_plan.auto_sharding_objective

        # Read sharding specs
        logical_mesh_shape = stage_plan.logical_mesh_shape
        accumulate_grad_in_avals = [
            avals[i] for i in accumulate_grad_invar_indices
        ] + grad_avals
        apply_grad_in_avals = \
            [avals[i] for i in apply_grad_invar_indices] + grad_avals
        accumulate_grad_input_sharding_specs, grad_sharding_specs = (
            get_input_output_sharding_specs(accumulate_grad.get_module(),
                                            accumulate_grad_in_avals,
                                            grad_avals,
                                            physical_mesh.num_devices,
                                            logical_mesh_shape))
        apply_grad_input_sharding_specs, output_sharding_specs = (
            get_input_output_sharding_specs(apply_grad.get_module(),
                                            apply_grad_in_avals, out_avals,
                                            physical_mesh.num_devices,
                                            logical_mesh_shape))
        self.output_sharding_specs = output_sharding_specs
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
                new_shape = (num_micro_batches *
                             aval.shape[0],) + aval.shape[1:]
                new_spec = get_microbatch_sharding_spec(
                    global_arg_sharding_specs[i], batch_dim, num_micro_batches)
                global_arg_shard_indices.append(
                    pxla.spec_to_indices(new_shape, new_spec))
            else:
                global_arg_shard_indices.append(
                    pxla.spec_to_indices(aval.shape,
                                         global_arg_sharding_specs[i]))

        accumulate_grad_batch_arg_indices = [
            i for i, j in enumerate(accumulate_grad_invar_indices)
            if batch_invars[j]
        ]
        grad_shard_shapes = [
            get_shard_shape(aval, spec)
            for aval, spec in zip(grad_avals, grad_sharding_specs)
        ]
        grad_shard_dtypes = [aval.dtype for aval in grad_avals]
        self.global_arg_sharding_specs = global_arg_sharding_specs
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
                    accumulate_grad, apply_grad, accumulate_grad_invar_indices,
                    apply_grad_invar_indices, accumulate_grad_batch_arg_indices,
                    grad_shard_shapes, grad_shard_dtypes, stage_plan,
                    donated_invars, batch_invars, num_grads, num_micro_batches)
            # The following members will be fetched from the workers later
            self.fully_optimized_hlo_text = None
            self.grad_sync_channel_ids = None
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            backend = physical_mesh.backend

            self.accumulate_grad = run_backend_compilation(
                backend, accumulate_grad, stage_plan, physical_mesh.num_devices)
            self.apply_grad = run_backend_compilation(backend, apply_grad,
                                                      stage_plan,
                                                      physical_mesh.num_devices)
            self.allocate_zero_buffers = compile_allocate_zero_buffers(
                backend, physical_mesh.num_devices, grad_shard_shapes,
                grad_shard_dtypes)
            self.accumulate_grad_batch_arg_indices = (
                accumulate_grad_batch_arg_indices)

            self.fully_optimized_hlo_text = (
                self.accumulate_grad.hlo_modules()[0].to_string() +
                self.apply_grad.hlo_modules()[0].to_string())
            self.grad_sync_channel_ids = get_grad_sync_channel_ids(
                self.accumulate_grad.hlo_modules()[0])
            self.skip_allreduce_env_name = (
                self.accumulate_grad.hlo_modules()[0].name +
                "XLA_SKIP_NCCL_COLLECTIVE_IDS")

        # Set up timers
        self.exec_timer_name = get_execution_timer_name(self.exec_uuid)
        self.shard_args_timer_name = self.exec_timer_name + "-shard-args"
        self.sync_func = get_sync_func_driver(physical_mesh)

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        num_micro_batches = self.num_micro_batches
        grad_avals = self.grad_avals
        num_grads = len(grad_avals)
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_outs = len(self.out_avals)

        timers(self.shard_args_timer_name).start()
        input_bufs = physical_mesh.shard_args_to_bufs(
            self.global_arg_shard_indices, self.donated_invars,
            self.batch_invars, num_micro_batches, args)

        first_batch_bufs = input_bufs
        next_batches_bufs = []
        for i in self.global_batch_arg_indices:
            micro_batches = input_bufs[i]
            first_batch_bufs[i] = micro_batches[0]
            next_batches_bufs.extend(micro_batches[1:])
        timers(self.shard_args_timer_name).stop()

        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            first_batch_uuids = np.array([ref.uuid for ref in first_batch_bufs])

            if next_batches_bufs:
                next_batches_uuids = np.array(
                    [ref.uuid for ref in next_batches_bufs])
            else:
                next_batches_uuids = (None,) * num_hosts

            output_uuids = next_array_uuids(num_outs)

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, first_batch_uuids, next_batches_uuids,
                    output_uuids, global_config.shard_parallel_sync_for_timer,
                    global_config.shard_parallel_sync_for_timer)

            # Gather output buffers
            output_bufs = np.array(
                [RemoteArrayRef(physical_mesh, uuid) for uuid in output_uuids])

            # Mark donated input buffers as already deleted on workers.
            for ary_ref, is_donated in zip(first_batch_bufs,
                                           self.donated_invars):
                if is_donated:
                    ary_ref.set_deleted_on_workers()

            # Mark micro batch buffers as already deleted on workers.
            for ary_ref in next_batches_bufs:
                ary_ref.set_deleted_on_workers()
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            sync_func = (self.sync_func if
                         global_config.shard_parallel_sync_for_timer else None)

            # Prepare gradient buffers
            timers(self.exec_timer_name).start(sync_func)
            grad_bufs = (
                self.allocate_zero_buffers.execute_sharded_on_local_devices([]))

            # Call accumulate_grad multiple times
            tmp_input_bufs = ([
                first_batch_bufs[i] for i in self.accumulate_grad_invar_indices
            ] + grad_bufs)
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
                grad_bufs = (self.accumulate_grad.
                             execute_sharded_on_local_devices(tmp_input_bufs))

            # Call apply_grad
            tmp_input_bufs = (
                [first_batch_bufs[i] for i in self.apply_grad_invar_indices] +
                grad_bufs)
            output_bufs = self.apply_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)
            timers(self.exec_timer_name).stop(sync_func)

        # Wrap output buffers as ShardedArray
        return self.outs_handler(output_bufs)

    def get_input_placement_specs(self):
        """
        Return the preferred placement specs for input arguments.
        The return value is a pytree of PlacementSpec
        with the same structure as the input pytree.
        """
        return wrap_to_placement_spec_tree(self.physical_mesh, self.avals,
                                           self.global_arg_sharding_specs,
                                           self.in_tree)

    def get_output_placement_specs(self):
        """
        Return the preferred placement specs for outputs.
        The return value is a pytree of PlacementSpec
        with the same structure as the output pytree.
        """
        return wrap_to_placement_spec_tree(self.physical_mesh, self.out_avals,
                                           self.output_sharding_specs,
                                           self.out_tree)

    def get_parallel_plan(self):
        """Get the overall parallel plan."""
        cluster_info = ClusterInfo(self.physical_mesh.num_hosts,
                                   self.physical_mesh.num_devices_per_host)
        return ParallelPlan(cluster_info, self.num_micro_batches,
                            self.auto_sharding_option, None,
                            tree_leaves(self.get_input_placement_specs()))

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
        if isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh):
            return ray.get(self.physical_mesh.workers[0].
                           get_exec_total_allocation_size.remote(
                               self.exec_uuid))
        else:
            assert isinstance(self.physical_mesh, LocalPhysicalDeviceMesh)
            return max(self.accumulate_grad.total_allocation_size(),
                       self.apply_grad.total_allocation_size())

    def get_hlo_text(self, status: HloStatus = HloStatus.FULLY_OPTIMIZED):
        """Return the HLO IR in the text format."""
        if status == HloStatus.FULLY_OPTIMIZED:
            if self.fully_optimized_hlo_text is not None:
                return self.fully_optimized_hlo_text
            assert isinstance(self.physical_mesh, DistributedPhysicalDeviceMesh)
            self.fully_optimized_hlo_text = ray.get(
                self.physical_mesh.workers[0].get_exec_hlo_text.remote(
                    self.exec_uuid))
            self.grad_sync_channel_ids = ray.get(
                self.physical_mesh.workers[0].get_exec_grad_sync_channel_ids.
                remote(self.exec_uuid))
            return self.fully_optimized_hlo_text
        else:
            raise ValueError(f"Invalid status: {status}")

    def dump_debug_info(self, folder: str):
        """
        Dump intermediate representations and other informations for debugging.
        """
        os.makedirs(folder, exist_ok=True)
        name = self.accumulate_grad_hlo.name
        name = name[:name.index("shard_parallel") - 1]
        prefix = os.path.join(folder, name)
        with open(f"{prefix}.hlo", "w") as f:
            f.write(self.get_hlo_text())
        with open(f"{prefix}.grad_sync_channel_ids.txt", "w") as f:
            f.write(str(self.grad_sync_channel_ids) + "\n")
        with open(f"{prefix}.mem_usage.txt", "w") as f:
            f.write(f"total_allocation_size: "
                    f"{self.get_total_allocation_size()/(1024**3):.3f} GB\n")
        with open(f"{prefix}_input_placement_specs.txt", "w") as f:
            f.write(str(self.get_input_placement_specs()) + "\n\n")
        with open(f"{prefix}_output_placement_specs.txt", "w") as f:
            f.write(str(self.get_output_placement_specs()) + "\n\n")


class GradAccMeshWorkerExecutable(MeshWorkerExecutable):
    """The worker part of a gradient accumulation mesh executable."""

    def __init__(self, worker: "MeshHostWorker", uuid: int,
                 accumulate_grad: WrappedHlo, apply_grad: WrappedHlo,
                 accumulate_grad_invar_indices: Sequence[int],
                 apply_grad_invar_indices: Sequence[int],
                 accumulate_grad_batch_arg_indices: Sequence[int],
                 grad_shard_shapes: Sequence[Sequence[int]],
                 grad_shard_dtypes: Sequence[jnp.dtype], stage_plan: StagePlan,
                 donated_invars: Sequence[bool], batch_invars: Sequence[bool],
                 num_grads: int, num_micro_batches: int):
        num_devices = np.prod(stage_plan.logical_mesh_shape)
        assert num_devices == len(worker.backend.devices())

        self.accumulate_grad = run_backend_compilation(worker.backend,
                                                       accumulate_grad,
                                                       stage_plan, num_devices)
        self.apply_grad = run_backend_compilation(worker.backend, apply_grad,
                                                  stage_plan, num_devices)
        self.allocate_zero_buffers = compile_allocate_zero_buffers(
            worker.backend, num_devices, grad_shard_shapes, grad_shard_dtypes)
        self.accumulate_grad_invar_indices = accumulate_grad_invar_indices
        self.apply_grad_invar_indices = apply_grad_invar_indices
        self.accumulate_grad_batch_arg_indices = (
            accumulate_grad_batch_arg_indices)
        self.donated_invars = donated_invars
        self.batch_invars = batch_invars
        self.num_grads = num_grads
        self.num_micro_batches = num_micro_batches
        self.buffer_dict = worker.buffers
        self.grad_sync_channel_ids = get_grad_sync_channel_ids(
            self.accumulate_grad.hlo_modules()[0])
        self.skip_allreduce_env_name = (
            self.accumulate_grad.hlo_modules()[0].name +
            "XLA_SKIP_NCCL_COLLECTIVE_IDS")

        # Set up timers
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self, first_batch_uuids: Sequence[int],
                          next_batches_uuids: Sequence[int],
                          output_uuids: Sequence[int], sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.buffer_dict
        num_micro_batches = self.num_micro_batches

        tmp_input_bufs = [
            buffer_dict[first_batch_uuids[i]]
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
                    tmp_input_bufs[idx] = buffer_dict[next_batches_uuids[
                        j * (num_micro_batches - 1) + (i - 1)]]
            if i == num_micro_batches - 1:
                os.environ[self.skip_allreduce_env_name] = ""
            grad_bufs = self.accumulate_grad.execute_sharded_on_local_devices(
                tmp_input_bufs)

        # Call apply_grad
        tmp_input_bufs = [
            buffer_dict[first_batch_uuids[i]]
            for i in self.apply_grad_invar_indices
        ] + grad_bufs
        output_bufs = self.apply_grad.execute_sharded_on_local_devices(
            tmp_input_bufs)
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

        # Store output buffers
        for i in range(len(output_uuids)):
            buffer_dict[output_uuids[i]] = output_bufs[i]

        # Delete donated input buffers
        delete_donated_buffers(buffer_dict, first_batch_uuids,
                               self.donated_invars)

        # Delete micro batch buffers
        if next_batches_uuids is not None and \
                next_batches_uuids[0] is not None:
            for i in range(len(next_batches_uuids)):
                del buffer_dict[next_batches_uuids[i]]

    def get_hlo_text(self):
        return (self.accumulate_grad.hlo_modules()[0].to_string() +
                self.apply_grad.hlo_modules()[0].to_string())

    def get_total_allocation_size(self):
        """Get the total memory allocation size in bytes."""
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

    def __init__(self, physical_mesh: "PhysicalDeviceMesh", hlo: WrappedHlo,
                 stage_plan: StagePlan, avals: Sequence[ShapedArray],
                 out_avals: Sequence[ShapedArray],
                 donated_invars: Sequence[bool]):
        super().__init__(physical_mesh, hlo, stage_plan, avals, out_avals,
                         donated_invars)

    def _set_executable(self, physical_mesh, hlo, stage_plan):
        """Put the executable on workers."""
        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            for w in physical_mesh.workers:
                w.put_executable.remote(self.exec_uuid,
                                        PartialGradAccMeshWorkerExecutable, hlo,
                                        stage_plan, self.donated_invars)
            self.hlo_text = None  # will be fetched from the workers later
            self.grad_sync_channel_ids = None
            self.skip_allreduce_env_name = None
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            self.compiled = run_backend_compilation(physical_mesh.backend, hlo,
                                                    stage_plan,
                                                    physical_mesh.num_devices)
            self.hlo_text = self.compiled.hlo_modules()[0].to_string()
            self.grad_sync_channel_ids = get_grad_sync_channel_ids(
                self.compiled.hlo_modules()[0])
            self.skip_allreduce_env_name = (
                self.compiled.hlo_modules()[0].name +
                "XLA_SKIP_NCCL_COLLECTIVE_IDS")

    def launch_on_driver(self, *args, **kwargs):
        """Launch the executable on the driver."""
        assert "skip_grad_sync" in kwargs, (
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

    def __init__(self, worker: "MeshHostWorker", uuid: int, hlo: WrappedHlo,
                 stage_plan: StagePlan, donated_invars: Sequence[bool]):
        super().__init__(worker, uuid, hlo, stage_plan, donated_invars)
        self.grad_sync_channel_ids = get_grad_sync_channel_ids(
            self.compiled.hlo_modules()[0])
        self.skip_allreduce_env_name = (self.compiled.hlo_modules()[0].name +
                                        "XLA_SKIP_NCCL_COLLECTIVE_IDS")

    # pylint: disable=arguments-differ
    def execute_on_worker(self, input_uuids: Sequence[int],
                          output_uuids: Sequence[int], sync_before: bool,
                          sync_after: bool, skip_grad_sync: bool):
        """Run the executable on the worker."""
        os.environ[self.skip_allreduce_env_name] = (self.grad_sync_channel_ids
                                                    if skip_grad_sync else "")
        return super().execute_on_worker(input_uuids, output_uuids, sync_before,
                                         sync_after)

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
                physical_mesh.backend, physical_mesh.devices, grad_shard_shapes,
                grad_shard_dtypes)

        self.exec_timer_name = get_execution_timer_name(self.exec_uuid)
        self.sync_func = get_sync_func_driver(physical_mesh)

    def launch_on_driver(self, *args):
        """Launch the executable on the driver."""
        assert len(args) == 0, (
            f"allocate zero buffers does not need args, got {len(args)}")
        physical_mesh = self.physical_mesh
        num_hosts = physical_mesh.num_hosts
        num_outs = len(self.out_avals)

        if isinstance(physical_mesh, DistributedPhysicalDeviceMesh):
            # Get output uuids
            output_uuids = next_array_uuids(num_outs)

            # Execute SPMD binary
            for i in range(num_hosts):
                physical_mesh.workers[i].run_executable.remote(
                    self.exec_uuid, [], output_uuids)

            # Gather outputs
            output_bufs = np.array(
                [RemoteArrayRef(physical_mesh, uuid) for uuid in output_uuids])
        else:
            assert isinstance(physical_mesh, LocalPhysicalDeviceMesh)
            timers(self.exec_timer_name).start(self.sync_func)
            output_bufs = (
                self.allocate_zero_buffers.execute_sharded_on_local_devices([]))
            timers(self.exec_timer_name).stop(self.sync_func)

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

    def execute_on_worker(self, input_uuids: Sequence[int],
                          output_uuids: Sequence[int], sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        # pylint: disable=unused-argument
        buffer_dict = self.worker.buffers

        # Execute
        if global_config.enable_overlapping:
            xe.set_idx_to_uuid(output_uuids)
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        output_bufs = (
            self.allocate_zero_buffers.execute_sharded_on_local_devices([]))
        timers(self.timer_name).stop(self.sync_func if sync_after else None)
        for i in range(len(output_uuids)):
            buffer_dict[output_uuids[i]] = output_bufs[i]

    def __del__(self):
        self.allocate_zero_buffers.delete()


class UtilMeshWorkerExecutable(MeshWorkerExecutable):
    """Worker executable that runs a manually generated function. It is lighter
    than NormalMeshWorkerExecutable as it does not have a StagePlan.

    Currently, it is used for concatenate(will be deprecated after we move it
    to apply_grad) and allgather.
    """

    def __init__(self, worker, uuid, hlo: WrappedHlo):
        num_devices = len(worker.backend.devices())
        compile_options = get_compile_options(
            num_replicas=1,
            num_partitions=num_devices,
            device_assignment=np.arange(num_devices).reshape((1, -1)),
            use_spmd_partitioning=False,
            parameter_is_tupled_arguments=False,
            build_random_seed=global_config.compile_random_seed)
        xla_computation = hlo.get_computation()

        with XlaPassContext({
                "done-event::enable": global_config.enable_overlapping,
        }):
            self.exec = worker.backend.compile(xla_computation, compile_options)

        self.worker = worker
        self.timer_name = get_execution_timer_name(uuid)
        self.sync_func = get_sync_func_worker(worker)

    def execute_on_worker(self, input_uuids: Sequence[int],
                          output_uuids: Sequence[int], sync_before: bool,
                          sync_after: bool):
        """Run the executable on the worker."""
        buffer_dict = self.worker.buffers

        # Get input
        input_bufs = [buffer_dict[x] for x in input_uuids]

        if global_config.enable_overlapping:
            xe.computation_wait_events(input_uuids, self.worker.backend)
            xe.set_idx_to_uuid(output_uuids)

        # Execute
        timers(self.timer_name).start(self.sync_func if sync_before else None)
        output_bufs = self.exec.execute_sharded_on_local_devices(input_bufs)
        timers(self.timer_name).stop(self.sync_func if sync_after else None)

        for i in range(len(output_uuids)):
            buffer_dict[output_uuids[i]] = output_bufs[i]

    def __del__(self):
        self.exec.delete()


def get_index_select_mesh_executable(avals, sharding_specs, index, dim,
                                     device_mesh, donate_avals):
    if type(index) not in [ShapedArray, ShapeDtypeStruct]:
        index = xla.canonicalize_dtype(index)
    index_shape = xc.shape_from_pyval(index)
    key = hash(("index_select", tuple(avals), tuple(sharding_specs),
                tuple(donate_avals), dim, index_shape))
    if key in device_mesh.operation_executables:
        return device_mesh.operation_executables[key]
    index_aval = ShapedArray(index.shape, index.dtype)
    assert len(avals) == len(sharding_specs) == len(donate_avals)
    hlo = get_index_select_computation(sharding_specs, dim, avals, index_shape)
    hlo = run_spmd_partitioner_pass(hlo, device_mesh.num_devices)

    as_option = AutoShardingOption()
    strategy_config = StagePlan(global_config.compile_random_seed,
                                device_mesh.shape, 1 << 60,
                                as_option.all_reduce_threshold,
                                AutoShardingOption(), None, -1)
    out_tree = tree_flatten(avals)[1]
    executable = NormalMeshDriverExecutable(device_mesh,
                                            hlo,
                                            strategy_config,
                                            [*avals, index_aval],
                                            avals, [*donate_avals, False],
                                            out_tree=out_tree)
    device_mesh.operation_executables[key] = executable
    return executable
