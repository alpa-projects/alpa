import numpy as np
from typing import Dict, Sequence, Set, Tuple

import jax.numpy as jnp
from jax.core import ClosedJaxpr, Var, gensym, jaxpr_as_fun
from jax.interpreters import pxla

from parax.api import parallelize
from parax.device_mesh import DistributedArray, PhysicalDeviceMesh, VirtualMesh, _shard_device_array
from parax.global_env import global_config
from parax.pipeline_parallel.cross_mesh_resharding import (
    CollectiveGroup, ReshardingTask, ReshardingTaskSpec, VirtualDistributedArray
    as VDA)
from parax.pipeline_parallel.stage import JaxPipelineStage, merge_stages, rearrange_vars
from parax.pipeline_parallel.three_d_parallel import get_donation_mapping_and_modify


########################################
##### Profile tools
########################################
def split_global_use_and_donate(layers, layer_indices, donation_mapping,
                                global_outvars):
    '''
    Pick some layers(no need to be consecutive) and assume they are on a mesh, 
    this function then returns donation_mapping and global_use of each selected layer.
    Args:
        layers (Sequence[JaxPipelineStage]): all layers
        layer_indices (Set[int]): indices of selected layers, they are
        assumed to be in the same mesh
        donation_mapping (Dict[Var, Var]): known global donation mapping
        global_outvars (Sequence[Var]): global outvars
    Returns:
        donation_mapping: donation mapping of all picked layers
        global_used: a set of outvars used not only in selected layers
        layers: layers rearranged for donate invar
    '''
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    layer_indices = set(layer_indices)
    gensym_fn = gensym([layer.closed_jaxpr().jaxpr for layer in layers])
    num_layers = len(layers)
    out_donation_mapping = dict()
    out_global_used = set()
    used = set(global_outvars)
    local_used = set()  # limit donation
    new_layers = []
    for idx in reversed(range(num_layers)):
        layer = layers[idx]
        if idx in layer_indices:
            global_used = set()
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


def split_sharding_specs(layers: Sequence[JaxPipelineStage],
                         mixed_jaxpr: ClosedJaxpr, in_sharding_specs,
                         out_sharding_specs):
    '''
    Split sharding specs of layers. Some intermediate sharding specs are missed,
    but they do not cross mesh so this does not matter.
    '''

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


def compile_and_profile_layer_cost_c(layers: Sequence[JaxPipelineStage],
                                     mesh: PhysicalDeviceMesh,
                                     donation_mapping: Dict[Var, Var],
                                     global_used: Set[Var]):
    """
    Args:
        layers (Sequence[JaxPipelineStage]): forward and corresponding backward
        mesh (PhysicalDeviceMesh): the assigned mesh
        donation_mapping (Dict[Var, Var]): donation mapping of all selected layers
        global_used_list (Set[Var]): for each layer, record if each
            outvar is used outside the compiled layers
    """
    backup_config = global_config.backup()

    global_config.num_micro_batches = None
    global_config.devices = mesh
    global_config.strategy = "shard_parallel"
    global_config.use_dummy_value_for_benchmarking = True

    jaxprs = [layer.closed_jaxpr() for layer in layers]

    mixed_jaxpr = merge_stages(jaxprs, global_used, 'profile_tmp',
                               donation_mapping)
    donate_argnums = [
        idx for idx, var in enumerate(mixed_jaxpr.jaxpr.invars)
        if var in donation_mapping
    ]

    fn = jaxpr_as_fun(mixed_jaxpr)
    args = [
        jnp.zeros(v.aval.shape, v.aval.dtype) for v in mixed_jaxpr.jaxpr.invars
    ]
    executable = parallelize(
        fn, donate_argnums=donate_argnums).get_executable(*args)
    ret = executable.profile_with_dummy_inputs()

    global_config.restore(backup_config)
    split_in_specs, split_out_specs = split_sharding_specs(
        layers, mixed_jaxpr, executable.input_sharding_specs,
        executable.output_sharding_specs)
    return ret, split_in_specs, split_out_specs


def create_collective_group(src_mesh: PhysicalDeviceMesh,
                            dst_mesh: PhysicalDeviceMesh) -> CollectiveGroup:
    cg = CollectiveGroup(set(src_mesh.device_strs + dst_mesh.device_strs),
                         src_mesh, dst_mesh)
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


def profile_layer_cost_e(src: JaxPipelineStage, dst: JaxPipelineStage,
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
            task = ReshardingTask(task_spec, collective_group, val)
            tasks.append(task)

    for task in tasks:
        task.prepare_send_recv_tasks()
    src_phy_mesh.sync_workers()
    collective_group.dst_mesh.sync_workers()
    results = []
    for task in tasks:
        results.append(task.do_prepared(task.src_array, True))

    tot_cost = sum([max(result) for result in results])

    global_config.use_dummy_value_for_benchmarking = backup_use_dummy_value
    return tot_cost


########################################
##### Algorithm
########################################
def get_mesh_slicing_configs(
        grid: VirtualMesh, layers,
        B) -> Tuple[Sequence[np.ndarray], np.ndarray, Sequence[Sequence[int]]]:
    '''
    TODO(yonghao, zhuohan): mesh slicing and layer allocation algorithm
    Args:
        grid (VirtualMesh): the whole grid
        layers (Sequence[JaxPipelineStage]): clustered layers
        B (number of microbatches)
    Returns:
        configs (Sequence[np.ndarray]): mesh slicing configs of each solution
        costs (np.ndarray): cost of each solution
        solutions (Sequence[Sequence[int]]): solutions of layer assignment
            in form of a list recording the number of layers in each stage.
    '''
    pass


def config_to_logical_meshes(raw_mesh: VirtualMesh, config: np.ndarray):
    """
    Translate a config array into logical meshes
    Args:
        raw_mesh (VirtualMesh): the total mesh
        config (np.ndarray): how meshes are sliced. config[i][j] is the mesh for device(i, j)
    """
    mesh_info = []
    M = config.shape[0]
    N = config.shape[1]

    visited = set()
    max_num = -1
    for i in range(M):
        for j in range(N):
            if config[i][j] not in visited:
                mesh_num = config[i][j]
                visited.add(mesh_num)
                start = (i, j)
                for p in range(j, N):
                    if config[i][p] != mesh_num:
                        p -= 1
                        break
                for q in range(i, M):
                    if config[q][j] != mesh_num:
                        q -= 1
                        break
                end = (q, p)
                mesh_info.append((mesh_num, start, end))
                max_num = max(max_num, mesh_num)
    assert max_num >= 0
    meshes = (None for _ in range(max_num))
    for info in mesh_info:
        id, start, end = info
        meshes[id] = raw_mesh.slice(0, range(start[0], end[0] + 1)).slice(
            1, range(start[1], end[1] + 1))
    return meshes


def slice_mesh(layers, **kwargs):
    '''
    Args:
        layers (Sequence[JaxPipelineStage]): clustered layers
    Returns:
        layer_assignment: the assignment of layers
        sliced_meshes (Sequence[PhysicalDeviceMesh]): sliced physical meshes
    '''
    raw_mesh = global_config.devices
    B = global_config.num_micro_batches
    configs, costs, solutions = get_mesh_slicing_configs(raw_mesh, layers, B)
    best_idx = costs.argmax()[0]
    best_config = configs[best_idx]
    layer_assignment = solutions[best_idx]
    sliced_meshes = config_to_logical_meshes(raw_mesh, best_config)
    return layer_assignment, sliced_meshes