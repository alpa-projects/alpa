import gc
from typing import Dict, Sequence, Tuple

import tqdm
import numpy as np
import ray
from ray.util import ActorPool

import jax.numpy as jnp
from jax.core import ClosedJaxpr, Var, gensym, jaxpr_as_fun
from jax.interpreters import pxla
from jax.lib import xla_bridge, xla_client, xla_extension as _xla

from parax.device_mesh import DistributedArray, PhysicalDeviceMesh, VirtualMesh, _shard_device_array
from parax.global_env import global_config
from parax.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTask, ReshardingTaskSpec, VirtualDistributedArray
    as VDA)
from parax.pipeline_parallel.computation import (
    JaxPipelineComputation, get_donation_mapping_and_modify,
    merge_computation_jaxprs)
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                HloProtoStatus,
                                                sharding_proto_to_sharding_spec)
from parax.util import get_shard_shape, jaxpr_to_hlo_computation, OrderedSet
from parax.mesh_executable import PartialGradAccMeshDriverExecutable, ProtoAndSharding


class CompileWorker:
    """
    A ray actor to distributedly compile Jaxpr to HLO Proto.
    To activaite the worker, a gpu resource is required.
    """

    def __init__(self, global_config_backup):
        self.cnt = 0
        self.backend = xla_bridge.get_backend("gpu")
        # FIXME: global_config_backup is conflict with new_global_config
        #        in compile_stage_with_search.
        global_config.restore(global_config_backup)

    def _get_input_output_sharding(self, sharding_annotated_computation):
        hlo_module = sharding_annotated_computation.as_hlo_module()
        hlo_module.infer_spmd_shardings()
        input_shardings = hlo_module.spmd_parameters_shardings()
        output_sharding = hlo_module.spmd_output_sharding()
        input_sharding_protos = [
            sharding.proto_tuple() for sharding in input_shardings
        ]
        output_sharding_proto = output_sharding.proto_tuple()
        return input_sharding_protos, output_sharding_proto

    def compile_stage_with_search(self, new_global_config, logical_mesh, proto,
                                  avals, out_avals, donate_invars):
        """
        Compile a single stage with auto sharding.
        Args:
            new_global_config: the global config for compilation setting.
            logical_mesh: the logical mesh for compilation.
            proto: the proto of XlaComputation to be compiled
            avals: input avals
            out_avals: output avals
            donate_invars: donate invars of the computation to be compiled
        Returns:
            proto: The proto of compiled executable
            strategy_config: The sharding strategy from auto sharding
        """
        global_config.restore(new_global_config)
        self.cnt += 1

        jaxpr_config = (avals, out_avals, donate_invars)
        mesh_config = (None, [logical_mesh
                             ], global_config.mesh_shape_search_mode,
                       global_config.memory_budget_per_device, None, None)
        multiple_stage_config = {
            "multiple_stages": "stage_and_hooked",
            "grad_acc_num_micro_batches": None,
            "bypass_device_assignment_check": True
        }
        protos, hooked_proto, strategy_config = self.compile_with_config(
            proto, jaxpr_config, mesh_config, multiple_stage_config)
        assert len(protos) == 1, "Can only compile one stage"
        sharded_proto = protos[0]
        sharding_annotated_computation = xla_client.XlaComputation(
            sharded_proto)
        if logical_mesh.total_devices > 1:
            (input_sharding_protos, output_sharding_proto
            ) = self._get_input_output_sharding(sharding_annotated_computation)
        else:
            input_sharding_protos = None
            output_sharding_proto = None
        compiled = compile_with_given_strategy(
            self.backend,
            sharding_annotated_computation,
            strategy_config,
            logical_mesh.total_devices,
            bypass_device_assignment_check=True,
            hlo_proto_status=HloProtoStatus.SHARDING_ANNOTATED)
        optimized_proto = compiled.hlo_modules(
        )[0].as_serialized_hlo_module_proto()
        return (optimized_proto, strategy_config, input_sharding_protos,
                output_sharding_proto, hooked_proto)

    def compile_with_config(self, proto, jaxpr_config, mesh_config,
                            multiple_stage_config):
        built = xla_client.XlaComputation(proto)
        return compile_with_search(self.backend, built, *jaxpr_config,
                                   *mesh_config, **multiple_stage_config)


class CompileWorkerPool:
    """wrapped ray.util.ActorPool"""

    def __init__(self,
                 num_cpus,
                 num_gpus,
                 global_config_backup,
                 debug_mode=False):
        gpu_per_cpu = min(1, num_gpus / num_cpus * 0.5)
        worker_cls = ray.remote(num_cpus=1, num_gpus=gpu_per_cpu)(CompileWorker)
        global_config_backup.pop("devices")
        self.actors = [
            worker_cls.remote(global_config_backup) for _ in range(num_cpus)
        ]
        self.pool = ActorPool(self.actors)
        self.local_worker = CompileWorker() if debug_mode else None

    def local_get(self, fn, *value):
        return fn(self.local_worker, value)

    def submit(self, fn, value):
        self.pool.submit(fn, value)

    def get_next(self):
        return self.pool.get_next()

    def shutdown(self, force=True):
        for w in self.actors:
            if force:
                ray.kill(w)
            else:
                w.__ray_terminate__.remote()
        gc.collect()


class ProfileWorker:

    def __init__(self, virtual_mesh):
        self.mesh = virtual_mesh.get_physical_mesh()

    def profile(self, compiled_output, stage_info, intermediate_size):
        _, avals, out_avals, tot_donation = stage_info
        proto, config, in_shardings, out_shardings, hooked_proto = compiled_output
        compiled = ProtoAndSharding(proto=proto,
                                    input_shardings=in_shardings,
                                    output_shardings=out_shardings)
        donated_invars = (True,) * len(tot_donation) + (False,) * (
            len(avals) - len(tot_donation))
        executable = PartialGradAccMeshDriverExecutable(self.mesh, compiled,
                                                        config, avals,
                                                        out_avals,
                                                        donated_invars, [])
        cost = executable.profile_with_dummy_inputs(
            intermediates=intermediate_size)
        return cost


class ProfileWorkerPool:
    """wrapped ray.util.ActorPool"""

    def __init__(self, virtual_meshes):
        worker_cls = ray.remote(num_cpus=1e-3)(ProfileWorker)
        self.actors = [worker_cls.remote(mesh) for mesh in virtual_meshes]
        self.pool = ActorPool(self.actors)

    def submit(self, fn, value):
        self.pool.submit(fn, value)

    def get_next(self):
        return self.pool.get_next()

    def shutdown(self, force=True):
        for w in self.actors:
            if force:
                ray.kill(w)
            else:
                w.__ray_terminate__.remote()
        gc.collect()


def split_global_use_and_donate(layers, layer_indices, donation_mapping,
                                global_outvars):
    '''
    Pick some layers(no need to be consecutive) and assume they are on a mesh,
    this function then returns donation_mapping and global_use of each selected layer.
    Args:
        layers (Sequence[JaxPipelineComputation]): all layers
        layer_indices (OrderedSet[int]): indices of selected layers, they are
        assumed to be in the same mesh
        donation_mapping (Dict[Var, Var]): known global donation mapping
        global_outvars (Sequence[Var]): global outvars
    Returns:
        donation_mapping: donation mapping of all picked layers
        global_used: an OrderedSet of outvars used not only in selected layers
        layers: layers rearranged for donate invar
    '''
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    layer_indices = OrderedSet(layer_indices)
    gensym_fn = gensym([layer.closed_jaxpr().jaxpr for layer in layers])
    num_layers = len(layers)
    out_donation_mapping = dict()
    out_global_used = OrderedSet()
    used = OrderedSet(global_outvars)
    local_used = OrderedSet()  # limit donation
    new_layers = []
    for idx in reversed(range(num_layers)):
        layer = layers[idx]
        if idx in layer_indices:
            global_used = OrderedSet()
            local_donation, new_layer = get_donation_mapping_and_modify(
                layer, reversed_donation_mapping, gensym_fn)
            for invar in local_donation.keys():
                assert invar not in global_used and invar not in local_used

            global_used = [var for var in new_layer.outvars if var in used]
            out_donation_mapping.update(local_donation)
            out_global_used.update(global_used)
            local_used.update(new_layer.invars)
            new_layers.append(new_layer)
            continue
        used.update(layer.invars)
    new_layers = list(reversed(new_layers))
    return out_donation_mapping, out_global_used, new_layers


def split_sharding_specs(layers: Sequence[JaxPipelineComputation],
                         mixed_jaxpr: ClosedJaxpr, in_sharding_specs,
                         out_sharding_specs):
    """
    Split sharding specs of layers. Some intermediate sharding specs are missed,
    but they do not cross mesh so this does not matter.
    """

    in_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.invars, in_sharding_specs))
    out_sharding_dict = dict(zip(mixed_jaxpr.jaxpr.outvars, out_sharding_specs))
    layer_in_sharding_specs = []
    layer_out_sharding_specs = []
    for layer in layers:
        layer_in_sharding_specs.append(
            [in_sharding_dict.get(var, None) for var in layer.invars])
        layer_out_sharding_specs.append(
            [out_sharding_dict.get(var, None) for var in layer.outvars])
    return layer_in_sharding_specs, layer_out_sharding_specs


def compile_and_profile_stage_compute_cost(
        layers: Sequence[JaxPipelineComputation], mesh: PhysicalDeviceMesh,
        donation_mapping: Dict[Var, Var], global_used: OrderedSet[Var]):
    """
    Args:
        layers (Sequence[JaxPipelineComputation]): forward and corresponding backward
        mesh (PhysicalDeviceMesh): the assigned mesh
        donation_mapping (Dict[Var, Var]): donation mapping of all selected layers
        global_used_list (OrderedSet[Var]): for each layer, record if each
            outvar is used outside the compiled layers
    """
    backup_config = global_config.backup()

    global_config.num_micro_batches = None
    global_config.devices = mesh
    global_config.strategy = "shard_parallel"
    global_config.use_dummy_value_for_benchmarking = True

    jaxprs = [layer.closed_jaxpr() for layer in layers]

    mixed_jaxpr = merge_computation_jaxprs(jaxprs, global_used, "profile_tmp",
                                           donation_mapping)
    donate_argnums = [
        idx for idx, var in enumerate(mixed_jaxpr.jaxpr.invars)
        if var in donation_mapping
    ]

    fn = jaxpr_as_fun(mixed_jaxpr)
    args = [
        jnp.zeros(v.aval.shape, v.aval.dtype) for v in mixed_jaxpr.jaxpr.invars
    ]
    from parax.api import parallelize
    executable = parallelize(
        fn, donate_argnums=donate_argnums).get_executable(*args)
    ret = executable.profile_with_dummy_inputs()

    global_config.restore(backup_config)
    split_in_specs, split_out_specs = split_sharding_specs(
        layers, mixed_jaxpr, executable.input_sharding_specs,
        executable.output_sharding_specs)
    return ret, split_in_specs, split_out_specs


def generate_stage_info(all_layers,
                        selected_indices,
                        donation_mapping,
                        global_outvars,
                        name,
                        insert_hook_after=None):
    """Combine selected layers together for profiling"""
    backend = xla_bridge.get_backend("gpu")

    # TODO(yonghao): infer used_outside etc. in batches
    selected_donation_mapping, used_outside, layers = split_global_use_and_donate(
        all_layers, selected_indices, donation_mapping, global_outvars)

    jaxprs = [layer.closed_jaxpr() for layer in layers]

    merged, hook = merge_computation_jaxprs(jaxprs, used_outside, "0",
                                            selected_donation_mapping,
                                            insert_hook_after)
    outvars = OrderedSet(merged.jaxpr.outvars)
    avals = [var.aval for var in merged.jaxpr.invars]
    out_avals = [var.aval for var in merged.jaxpr.outvars]
    tot_donation = [
        invar in selected_donation_mapping and
        selected_donation_mapping[invar] in outvars
        for invar in merged.jaxpr.invars
    ]

    built = jaxpr_to_hlo_computation(name, merged, tot_donation, backend)
    proto = built.as_serialized_hlo_module_proto()
    return (proto, avals, out_avals, tot_donation), hook


def compile_all(stage_info_list, logical_mesh: VirtualMesh, num_cpus, num_gpus):
    """
    Args:
        stage_info_list: List of info for compilation. Each info is a tuple with:
            (proto, in_avals, out_avals, donate_invars)
    """
    compile_workers = CompileWorkerPool(num_cpus, num_gpus,
                                        global_config.backup())
    backup_config = global_config.backup()
    global_config.num_micro_batches = None
    global_config.devices = logical_mesh
    global_config.strategy = "shard_parallel"
    global_config.use_dummy_value_for_benchmarking = True
    compile_config = global_config.backup()
    for stage_info in stage_info_list:
        proto, avals, out_avals, donate_invars = stage_info
        compile_workers.submit(
            lambda w, v: w.compile_stage_with_search.remote(*v),
            (compile_config, logical_mesh, proto, avals, out_avals,
             donate_invars))

    compiled_outputs = []
    for stage_info in tqdm.tqdm(stage_info_list):
        compiled_output = compile_workers.get_next()
        compiled_outputs.append(compiled_output)

    compile_workers.shutdown()
    global_config.restore(backup_config)
    return compiled_outputs


def create_collective_group(src_mesh: PhysicalDeviceMesh,
                            dst_mesh: PhysicalDeviceMesh) -> CollectiveGroup:
    cg = CollectiveGroup(
        OrderedSet(src_mesh.device_strs + dst_mesh.device_strs), src_mesh,
        dst_mesh)
    cg.instantiate()
    return cg


def dummy_resharding_strategy(spec: ReshardingTaskSpec):
    strategy = []
    _sender_loads = {sender: 0 for sender in spec.src.device_mesh.device_strs}
    for dst_tile, src_tileslices, _ in spec.dst_tile_to_src_tiles_map:
        # plan is a 2D array
        per_spec_plan = np.empty(
            (len(dst_tile.replica_device_strs), len(src_tileslices)),
            dtype=object)
        for receiver_idx, _ in enumerate(dst_tile.replica_device_strs):
            for src_tileslice_idx, src_tileslice in enumerate(src_tileslices):
                loads = {
                    sender: _sender_loads[sender]
                    for sender in src_tileslice.replica_device_strs
                }
                sender = min(loads, key=loads.get)
                per_spec_plan[receiver_idx][src_tileslice_idx] = sender
                # upload load on-the-fly
                _sender_loads[sender] += src_tileslice.slice_size
        strategy.append(per_spec_plan)
    spec.set_resharding_strategy(strategy)
    return strategy


def profile_layer_communication_cost(
        src: JaxPipelineComputation, dst: JaxPipelineComputation,
        src_outvar_sharding_spec, dst_invar_sharding_spec,
        src_mesh: VirtualMesh, dst_mesh: VirtualMesh,
        collective_group: CollectiveGroup):
    src_outvars = {v: idx for idx, v in enumerate(src.outvars)}
    tot_cost = 0
    backup_use_dummy_value = global_config.use_dummy_value_for_benchmarking
    global_config.use_dummy_value_for_benchmarking = True
    tasks = []
    src_phy_mesh = collective_group.src_mesh
    for idx, invar in enumerate(dst.invars):
        if invar in src_outvars:
            out_sharding_spec = src_outvar_sharding_spec[src_outvars[invar]]
            in_sharding_spec = dst_invar_sharding_spec[idx]
            src_array = VDA(device_mesh=src_mesh,
                            aval=invar.aval,
                            sharding_spec=out_sharding_spec)
            dst_array = VDA(device_mesh=dst_mesh,
                            aval=invar.aval,
                            sharding_spec=in_sharding_spec)
            task_spec = ReshardingTaskSpec(src_array, dst_array)
            # create resharding strategy, ignore global load balance
            dummy_resharding_strategy(task_spec)
            # create distributed array as dummy inputs
            input_indices = pxla.spec_to_indices(invar.aval.shape,
                                                 out_sharding_spec)
            remote_buffers = _shard_device_array(jnp.zeros_like(invar.aval),
                                                 src_phy_mesh, input_indices)
            val = DistributedArray(src_phy_mesh, invar.aval, in_sharding_spec,
                                   remote_buffers, input_indices)
            task = ReshardingTask(task_spec, collective_group,
                                  collective_group.src_mesh,
                                  collective_group.dst_mesh)
            tasks.append(task)

    for task in tasks:
        task.put_send_recv_tasks()
    src_phy_mesh.sync_workers()
    collective_group.dst_mesh.sync_workers()
    results = []
    for task in tasks:
        results.append(task.do_prepared(task.src_array, True))

    tot_cost = sum([max(result) for result in results])

    global_config.use_dummy_value_for_benchmarking = backup_use_dummy_value
    return tot_cost


def compute_intermediate_size(serialized_proto, hook, config):
    """Compute bytes of serialized proto"""

    def get_byte(aval):
        return np.prod(aval.shape) * np.dtype(aval.dtype).itemsize

    avals = [v.aval for v in hook.invars]
    logical_mesh_shape = config.logical_mesh_shape
    if np.prod(logical_mesh_shape) == 1:
        tot = sum([get_byte(aval) for aval in avals])
        return tot
    hlo_sharding = _xla.HloSharding(serialized_proto[0]).proto_tuple()
    assert len(hlo_sharding[3]) == len(hook.invars), hlo_sharding
    sharding_specs = sharding_proto_to_sharding_spec(hlo_sharding, avals,
                                                     logical_mesh_shape)
    sharded_shapes = [
        get_shard_shape(aval, spec)
        for aval, spec in zip(avals, sharding_specs)
    ]
    tot = sum([
        np.prod(shape) * np.dtype(aval.dtype).itemsize
        for shape, aval in zip(sharded_shapes, avals)
    ])
    return tot
