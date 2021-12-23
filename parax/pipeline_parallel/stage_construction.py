from datetime import datetime
import math
from time import time
from typing import Sequence

import numba
import numpy as np
import ray
import tqdm

from parax.device_mesh import DeviceCluster, VirtualPhysicalMesh
from parax.pipeline_parallel.automatic_layer_slicing import eqn_flops
from parax.pipeline_parallel.computation import (JaxPipelineComputation,
                                                 merge_computation_jaxprs)
from parax.pipeline_parallel.stage_profiling import (
    compute_apply_grad_invar_size, compute_intermediate_size,
    split_global_use_and_donate, generate_stage_info, compile_all, profile_all,
    ProfileWorkerPool)
from parax.timer import timers
from parax.util import OrderedSet
from parax.global_env import global_config

last_compute_cost_file_name = None
last_forward_stage_layer_ids = None
last_submesh_shapes = None
last_logical_mesh_shapes = None
last_autosharding_global_configs = None


def get_last_dp_result():
    global last_compute_cost_file_name
    global last_forward_stage_layer_ids, last_submesh_shapes
    global last_logical_mesh_shapes, last_autosharding_global_configs
    return (last_compute_cost_file_name, last_forward_stage_layer_ids,
            last_submesh_shapes, last_logical_mesh_shapes,
            last_autosharding_global_configs)


@numba.jit(nopython=True)
def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
            num_autosharding_configs, compute_cost, max_n_succ_stages,
            max_stage_cost):
    # For f, layer ID start from 0
    # f[#pipeline stages,
    #   layer id that is currently being considered,
    #   number of devices used]
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0] = 0
    for s in range(1, num_layers + 1):
        for i in range(num_layers - 1, -1, -1):
            for j in range(1, num_devices + 1):
                for k in range(num_layers, i, -1):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            # TODO(zhuohan): This level of for loop is not
                            #   necessary. It can be optimized by sorting
                            #   the logical mesh shapes.
                            for l in range(num_autosharding_configs):
                                if s - 1 <= max_n_succ_stages[i, k - 1, m, l]:
                                    stage_cost = compute_cost[i, k - 1, m, l]
                                    new_cost = f[s - 1, k, j -
                                                 n_submesh_devices] + stage_cost
                                    if (stage_cost <= max_stage_cost and
                                            new_cost < f[s, i, j]):
                                        f[s, i, j] = new_cost
                                        f_stage_max[s, i, j] = max(
                                            f_stage_max[s - 1, k,
                                                        j - n_submesh_devices],
                                            stage_cost)
                                        f_argmin[s, i, j] = (k, m, l)

    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, 0, num_devices] < best_total_cost:
            best_s = s
            best_total_cost = f[s, 0, num_devices]

    if np.isinf(best_total_cost):
        return np.inf, None

    total_cost = f[best_s, 0, num_devices] + (
        num_microbatches - 1) * f_stage_max[best_s, 0, num_devices]
    current_s = best_s
    current_layer = 0
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer < num_layers and current_devices > 0:
        next_start_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices])
        assert next_start_layer != -1 and current_devices != -1
        res.append(((current_layer, next_start_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == num_layers and
            current_devices == 0)

    return total_cost, res


def dp(num_layers, num_devices, num_microbatches, submesh_choices,
       num_autosharding_configs, compute_cost, max_n_succ_stages):
    timers("stage-construction-dp").start()

    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl(num_layers, num_devices, num_microbatches,
                                 submesh_choices, num_autosharding_configs,
                                 compute_cost, max_n_succ_stages,
                                 max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
    assert best_solution is not None, "no solution in auto stage construction."

    timers("stage-construction-dp").suspend()
    return best_cost, best_solution


def get_submesh_choices(mesh: VirtualPhysicalMesh):
    if global_config.fix_physical_mesh_shape:
        return [global_config.fix_physical_mesh_shape]
    num_hosts = mesh.num_hosts
    num_devices_per_host = mesh.num_devices_per_host
    submesh_choices = []

    # smaller submeshes:
    i = 1
    while i <= num_devices_per_host:
        submesh_choices.append((1, i))
        i *= 2
    assert submesh_choices[-1][1] == num_devices_per_host, (
        "Only supports the cases where num_devices_per_host is power of two, "
        "while now num_devices_per_host = {}".format(num_devices_per_host))

    # larger meshes:
    if global_config.submesh_choices_mode == "all":
        for i in range(2, num_hosts + 1):
            submesh_choices.append((i, num_devices_per_host))
    elif global_config.submesh_choices_mode == "power_of_two":
        i = 2
        while i <= num_hosts:
            submesh_choices.append((i, num_devices_per_host))
            i *= 2
    else:
        raise ValueError("Invalid submesh_choices: {}".format(
            global_config.submesh_choices))

    return tuple(submesh_choices)


def get_one_submesh_autosharding_config_choices(virtual_submesh, option,
                                                batch_size):
    """
    Return a list of logical meshes and autosharding configs for the
    auto stage construction algorithm.

    Args:
        option (string): ["all", "single_node_model_parallel", "default"].
    """
    results = []
    num_devices = virtual_submesh.total_devices
    if option in ["all", "single_node_model_parallel"]:
        if option == "all":
            max_mp_dimension = num_devices
        else:  # option == "single_node_model_parallel"
            max_mp_dimension = virtual_submesh.num_devices_per_host

        for mp_size in range(1, max_mp_dimension + 1):
            if num_devices % mp_size == 0:
                dp_size = num_devices // mp_size
                if batch_size % dp_size == 0:
                    results.append((virtual_submesh.get_logical_mesh(
                        (dp_size, mp_size)), {
                            "force_batch_dim_to_mesh_dim": 0
                        }))
        results.append((virtual_submesh.get_logical_mesh((num_devices, 1)), {}))
    elif option == "default":
        results.append((virtual_submesh.get_default_logical_mesh(), {}))
    elif option == "only_dp":
        results.append((virtual_submesh.get_logical_mesh((num_devices, 1)), {
            "force_batch_dim_to_mesh_dim": 0
        }))
    return results


def get_all_submesh_autosharding_config_choices(virtual_mesh, submesh_choices,
                                                option, batch_size):
    # a config is: (logical_mesh, autosharding_global_configs)
    # each (2D Mesh with force batch dim) + (1D Mesh with mix batch dim)
    autosharding_configs = []
    for submesh in submesh_choices:
        num_hosts, num_devices = submesh
        virtual_submesh = virtual_mesh.slice_2d(
            list(range(num_hosts)),
            [list(range(num_devices)) for _ in range(num_hosts)])
        submesh_autosharding_configs = (
            get_one_submesh_autosharding_config_choices(virtual_submesh, option,
                                                        batch_size))
        autosharding_configs.append(submesh_autosharding_configs)

    # Pad all submesh to the maximum number of configs
    max_num_autosharding_configs = max(
        [len(configs) for configs in autosharding_configs])
    for configs in autosharding_configs:
        configs += [None] * (max_num_autosharding_configs - len(configs))

    return autosharding_configs


def distributed_profile_on_mesh(meshes: Sequence[VirtualPhysicalMesh], layers,
                                donation_mapping, global_outvars,
                                apply_grad_layers, apply_grad_global_info,
                                autosharding_configs, cluster_size,
                                layer_flops_prefix_sum):
    timers("stage-construction-compilation").start()
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    tot_flops = layer_flops_prefix_sum[2 * num_layers]
    num_autosharding_configs = len(autosharding_configs)
    indices = list(range(2 * num_layers))
    stages = []

    print("- Generate all stage infos (Jaxpr -> HLO)")
    # TODO(yonghao): only generate these info once for all mesh shapes
    computation_source_ratio = meshes[0].total_devices / cluster_size
    is_full_mesh = computation_source_ratio == 1
    tolerance = global_config.auto_stage_construction_imbalance_tolerance
    for start in tqdm.tqdm(range(0, num_layers)):
        for end in tqdm.tqdm(range(start, num_layers), leave=False):
            if is_full_mesh and not (start == 0 and end == num_layers - 1):
                continue
            flops_ratio = (
                layer_flops_prefix_sum[end + 1] - layer_flops_prefix_sum[start]
                + layer_flops_prefix_sum[2 * num_layers - start] -
                layer_flops_prefix_sum[2 * num_layers - end - 1]) / tot_flops
            if ((computation_source_ratio > flops_ratio * (1 + tolerance)) or
                (computation_source_ratio < flops_ratio / (1 + tolerance))):
                continue
            layer_indices = (
                indices[start:end + 1] +
                indices[2 * num_layers - end - 1:2 * num_layers - start])
            selected_apply_grad_layers = [
                apply_grad_layers[idx] for idx in indices[start:end + 1]
            ]
            stage_name = "stage_{}_{}".format(start, end)
            (compile_info, intermediate_vars, profile_info,
             apply_info) = generate_stage_info(
                 layers,
                 layer_indices,
                 donation_mapping,
                 global_outvars,
                 stage_name,
                 insert_hook_after=end - start,
                 apply_grad_info=(selected_apply_grad_layers,
                                  *apply_grad_global_info))
            if is_full_mesh:
                intermediate_vars = []
            for config_idx, autosharding_config in enumerate(
                    autosharding_configs):
                if autosharding_config is not None:
                    stage_indices = (start, end, config_idx)
                    stages.append(
                        (stage_indices, compile_info, autosharding_config,
                         intermediate_vars, profile_info, apply_info))

    if len(stages) == 0:
        compute_cost = np.full(
            (num_layers, num_layers, num_autosharding_configs), np.inf)
        max_n_succ_stages = np.full(
            (num_layers, num_layers, num_autosharding_configs), -1)
        # Suspend timers
        timers("stage-construction-compilation").suspend()
        timers("stage-construction-profiling").start()
        timers("stage-construction-profiling").suspend()
        return compute_cost, max_n_succ_stages

    # TODO(zhuohan): set the number of workers as a tunable parameter
    print("- Compile all stages")
    compiled_outputs = compile_all(stages)
    timers("stage-construction-compilation").suspend()

    print("- Profile all stages")
    # shape of compute_cost and max_n_succ_stages:
    # (num_layers, num_layers, num_autosharding_configs)
    timers("stage-construction-profiling").start()
    compute_cost, max_n_succ_stages = profile_all(stages, compiled_outputs,
                                                  meshes, num_layers,
                                                  num_autosharding_configs)
    timers("stage-construction-profiling").suspend()
    return compute_cost, max_n_succ_stages


def _get_layer_flops_prefix_sum(layers):
    layer_flops_prefix_sum = [0]
    for layer in layers:
        layer_flops = sum([eqn_flops(eqn) for eqn in layer.eqns])
        layer_flops_prefix_sum.append(layer_flops_prefix_sum[-1] + layer_flops)
    return layer_flops_prefix_sum


def get_compute_cost(virtual_mesh: VirtualPhysicalMesh, submesh_choices,
                     autosharding_configs, layers, donation_mapping,
                     global_outvars, apply_grad_layers, apply_grad_global_info):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    num_autosharding_configs = len(autosharding_configs[0])
    cluster_size = virtual_mesh.total_devices
    layer_flops_prefix_sum = _get_layer_flops_prefix_sum(layers)

    compute_cost = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        np.inf)
    max_n_succ_stages = np.full(
        (num_layers, num_layers, num_submesh_choices, num_autosharding_configs),
        -1)
    print("-" * 20 + " Automatic stage clustering " + "-" * 20)
    print(f"submesh_choices: {submesh_choices}")

    # Reverse submesh_choices to test larger meshes first
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        print(f"- Profiling for submesh {mesh_id} {submesh}:")
        num_hosts, num_devices = submesh
        tic = time()
        if global_config.profile_with_whole_ray_cluster:
            whole_cluster_virtual_mesh = DeviceCluster(
            ).get_virtual_physical_mesh()
            sliced_virtual_meshes = (
                whole_cluster_virtual_mesh.slice_profiling_submeshes(
                    num_hosts, num_devices))
        else:
            sliced_virtual_meshes = virtual_mesh.slice_profiling_submeshes(
                num_hosts, num_devices)
        mesh_compute_cost, mesh_max_n_succ_stages = distributed_profile_on_mesh(
            sliced_virtual_meshes, layers, donation_mapping, global_outvars,
            apply_grad_layers, apply_grad_global_info,
            autosharding_configs[mesh_id], cluster_size, layer_flops_prefix_sum)

        compute_cost[:, :, mesh_id, :] = mesh_compute_cost
        max_n_succ_stages[:, :, mesh_id, :] = mesh_max_n_succ_stages
        toc = time()
        print(f'Profiling for submesh {mesh_id} {submesh} takes {toc - tic}'
              f' seconds')
        print(f'Profiled costs are: {mesh_compute_cost}')
        print(f'Profiled max_n_succ_stages are: {mesh_max_n_succ_stages}')
        print('-' * 50)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    compute_cost_file_name = (f"compute-cost-{timestamp}.npy")
    np.save(compute_cost_file_name, (compute_cost, max_n_succ_stages))
    global last_compute_cost_file_name
    last_compute_cost_file_name = compute_cost_file_name
    print(f'Compute cost saved to: {compute_cost_file_name}')
    print("-" * 70)
    return compute_cost, max_n_succ_stages


def get_sliced_virtual_submeshes(virtual_mesh, submeshe_shapes):
    num_hosts = virtual_mesh.num_hosts
    num_devices_per_host = virtual_mesh.num_devices_per_host
    submesh_sizes = [np.prod(submesh) for submesh in submeshe_shapes]
    virtual_submeshes = [None] * len(submeshe_shapes)
    assert sum(submesh_sizes) == virtual_mesh.total_devices
    sorted_submesh_indices = np.argsort(submesh_sizes)
    current_host_id = 0
    current_device_id = 0
    for i in reversed(sorted_submesh_indices):
        required_num_hosts, required_num_devices = submeshe_shapes[i]
        if required_num_devices == num_devices_per_host:
            assert current_device_id == 0
            assert current_host_id + required_num_hosts <= num_hosts, (
                "Do not have enough hosts for the solution.")
            virtual_submeshes[i] = virtual_mesh.slice_2d(
                range(current_host_id, current_host_id + required_num_hosts), [
                    range(num_devices_per_host)
                    for _ in range(required_num_hosts)
                ])
            current_host_id += required_num_hosts
        else:
            assert required_num_hosts == 1
            assert required_num_devices < num_devices_per_host
            assert (current_device_id + required_num_devices <=
                    num_devices_per_host), (
                        "Do not have enough devices in a host for the solution")
            virtual_submeshes[i] = virtual_mesh.slice_2d([current_host_id], [
                range(current_device_id,
                      current_device_id + required_num_devices)
            ])
            current_device_id += required_num_devices
            if current_device_id == num_devices_per_host:
                current_host_id += 1
                current_device_id = 0
    assert current_host_id == num_hosts
    assert current_device_id == 0
    return virtual_submeshes


def uniform_slice_mesh(original_mesh, num_meshes, submesh_shapes=None):
    """
    Slice the mesh uniformly.

    In this impl, we guarantee the slicing follows:
    - len(sliced_meshes) == num_stages / 2 (place forward/backward in a mesh);
    - higher priority to slice over the node dimension rather than gpu dimension
    (so to preserve nvlink usage).

    Args:
        original_mesh: a virtual device mesh.
        num_meshes: number of submeshes.
        submesh_shapes (List[Tuple(int, int)]): a list of desired submesh shapes.

    Returns:
        sliced_meshes (List[Mesh]): List of meshes to spawn worker on.
    """
    output_meshes = []
    if not original_mesh.is_distributed:
        raise RuntimeError("SingleDeviceMesh is not supported.")
    if original_mesh.total_devices < num_meshes:
        raise RuntimeError("#device < #workers.")
    num_device_per_mesh = int(original_mesh.total_devices / num_meshes)
    num_device_per_host = original_mesh.num_devices_per_host
    num_host = original_mesh.num_hosts

    if submesh_shapes == None:
        # uniformly slice the mesh by priority
        if num_device_per_host >= num_device_per_mesh:
            num_mesh_per_host = num_device_per_host // num_device_per_mesh
            for i in range(num_meshes):
                host_idx = i // num_mesh_per_host
                mesh_idx = i % num_mesh_per_host
                ind = list(range(num_device_per_host))
                mesh = original_mesh.slice_1d(0, [host_idx]).slice_1d(
                    1, [
                        ind[mesh_idx * num_device_per_mesh:(mesh_idx + 1) *
                            num_device_per_mesh]
                    ])
                output_meshes.append(mesh)
        else:
            num_host_per_mesh = math.ceil(num_device_per_mesh /
                                          num_device_per_host)
            ind = list(range(num_host))
            for i in range(num_meshes):
                output_meshes.append((original_mesh.slice_1d(
                    0, ind[num_host_per_mesh * i:num_host_per_mesh * (i + 1)])))
    else:
        num_required_host, num_required_device_per_host = submesh_shapes[0]
        assert num_required_host <= num_host, \
            "cannot satisfy physical mesh requirement, require {} hosts given {} hosts."\
                .format(num_required_host, num_host)
        assert num_required_device_per_host <= num_device_per_host, \
            "cannot satisfy physical mesh requirement, require {} gpus per host given {} gpus per host."\
                .format(num_required_device_per_host, num_device_per_host)
        # doing assignment
        if num_required_device_per_host == num_device_per_host:
            # allocate all devices of a host
            num_host_per_mesh = num_host // num_meshes
            output_meshes = [
                original_mesh.slice_1d(
                    0,
                    list(
                        range(i * num_host_per_mesh,
                              (i + 1) * num_host_per_mesh)))
                for i in range(num_meshes)
            ]
        else:
            assert num_device_per_host % num_required_device_per_host == 0
            cur_host_index = 0
            cur_device_index = 0
            for i in range(num_meshes):
                host_indices = list(
                    range(cur_host_index, cur_host_index + num_required_host))
                device_indices = list(
                    range(cur_device_index,
                          cur_device_index + num_required_device_per_host))
                device_indices = [device_indices] * len(host_indices)
                output_meshes.append(
                    original_mesh.slice_2d(host_indices, device_indices))
                # move the device in priority
                if cur_device_index + num_required_device_per_host == num_device_per_host:
                    cur_device_index = 0
                    cur_host_index = cur_host_index + num_required_host
                else:
                    cur_device_index = cur_device_index + num_required_device_per_host
            assert cur_host_index == num_host, "unable to satisfy the mesh requirement."
            assert cur_device_index == 0, "unable to satisfy the mesh requirement."
    return output_meshes


def cluster_layers_and_slice_mesh(layers,
                                  mesh,
                                  donation_mapping,
                                  global_outvars,
                                  num_micro_batches,
                                  batch_size,
                                  jax_apply_layers=None,
                                  apply_grad_global_info=None,
                                  pipeline_stage_mode="uniform_layer_gpipe",
                                  logical_mesh_search_space="default",
                                  cache_compute_cost=None,
                                  forward_stage_layer_ids=None,
                                  submesh_shapes=None,
                                  logical_mesh_shapes=None,
                                  autosharding_global_configs=None):
    """
    Cluster pipeline layers into stages, slice the device mesh
    into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        layers (Sequence[JaxPipelineComputation]): All the layers.
        mesh (VirtualPhysicalMesh): The cluser device mesh.
        donation_mapping: The donation_mapping for the layers.
        global_outvars: Global outvars of the layers.
        num_micro_batches: Number of microbatches for GPipe.
        pipeline_stage_mode (str): one of "auto_gpipe", "mannual_gpipe", "uniform_layer_gpipe".
        cache_compute_cost (Optional): Override the profiling results.
        forward_stage_layer_ids: hand-written layer-stage assignments.
        submesh_shapes (List): a list of allowed 2D mesh shapes.

    Returns:
        stage_layer_ids (List[List[int]]): The layer IDs of each stage.
        sliced_meshes (List[VirtualPhysicalMesh]): The shapes of all submeshes.
    """
    timers("stage-construction").start()

    if pipeline_stage_mode in ["auto_gpipe", "manual_gpipe"]:
        # Assume each forward layer corresponds to a backward layer
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2
        submesh_choices = get_submesh_choices(mesh)

        if pipeline_stage_mode == "auto_gpipe":
            autosharding_configs = get_all_submesh_autosharding_config_choices(
                mesh,
                submesh_choices,
                option=logical_mesh_search_space,
                batch_size=batch_size)
            num_autosharding_configs = len(autosharding_configs[0])

            # Use DP to find the optimal solution.
            if cache_compute_cost is not None:
                # FIXME(zhuohan): load max_n_succ_stages
                compute_cost, max_n_succ_stages = np.load(cache_compute_cost,
                                                          allow_pickle=True)
            else:
                compute_cost, max_n_succ_stages = get_compute_cost(
                    mesh, submesh_choices, autosharding_configs, layers,
                    donation_mapping, global_outvars, jax_apply_layers,
                    apply_grad_global_info)
            cost, solution = dp(num_layers, mesh.total_devices,
                                num_micro_batches, submesh_choices,
                                num_autosharding_configs, compute_cost,
                                max_n_succ_stages)

            # Parse solution
            forward_stage_layer_ids = [
                list(range(start_id, end_id))
                for (start_id, end_id), _, _ in solution
            ]
            submesh_shapes = [
                submesh_choices[submesh_id] for _, submesh_id, _ in solution
            ]
            selected_autosharding_configs = [
                autosharding_configs[submesh_id][autosharding_config_id]
                for _, submesh_id, autosharding_config_id in solution
            ]
            logical_mesh_shapes = [
                mesh.shape for mesh, _ in selected_autosharding_configs
            ]
            autosharding_global_configs = [
                config for _, config in selected_autosharding_configs
            ]

            # Print and store the results
            print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
            print("Result meshes:", submesh_shapes)
            print("Result logical_mesh_shapes:", logical_mesh_shapes)
            print("Result autosharding_global_configs:",
                  autosharding_global_configs)
            global last_forward_stage_layer_ids, last_submesh_shapes
            global last_logical_mesh_shapes, last_autosharding_global_configs
            last_forward_stage_layer_ids = forward_stage_layer_ids
            last_submesh_shapes = submesh_shapes
            last_logical_mesh_shapes = logical_mesh_shapes
            last_autosharding_global_configs = autosharding_global_configs
        elif pipeline_stage_mode == "manual_gpipe":
            # Manual-GPipe: use the user-provided solution
            # Sanity check that the user-provided solution is valid
            # Check forward_stage_layer_ids is a partition of range(num_layers)
            assert forward_stage_layer_ids is not None
            last_layer_id = 0
            for stage_layer_ids in forward_stage_layer_ids:
                for layer_id in stage_layer_ids:
                    assert layer_id == last_layer_id
                    last_layer_id += 1
            assert last_layer_id == num_layers
            # Check all the submesh_shapes are in submesh_choices
            assert submesh_shapes is not None
            for shape in submesh_shapes:
                assert shape in submesh_choices
        else:
            raise NotImplementedError("Unknown pipeline_stage_mode",
                                      pipeline_stage_mode)

        sliced_meshes = get_sliced_virtual_submeshes(mesh, submesh_shapes)
        num_forward_stages = len(forward_stage_layer_ids)
        backward_stage_layer_ids = [[
            2 * num_layers - 1 - i for i in reversed(layer_ids)
        ] for layer_ids in reversed(forward_stage_layer_ids)]
        stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages)) + list(
            reversed(range(num_forward_stages)))
        stage_outvars = get_stage_outvars(layers, stage_layer_ids,
                                          global_outvars)
        merged_stages = []
        for stage_id, layer_ids in enumerate(stage_layer_ids):
            stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
            stage_name = str(stage_id)
            merged_stage_jaxpr = merge_computation_jaxprs(
                stage_layer_jaxprs, stage_outvars[stage_id], stage_name,
                donation_mapping)
            merged_stage = JaxPipelineComputation.from_closed_jaxpr(
                stage_name, merged_stage_jaxpr)
            merged_stages.append(merged_stage)
        stages = merged_stages
    elif pipeline_stage_mode == "uniform_layer_gpipe":
        # This mode resembles Megatron in terms of the uniformity of mesh shapes.
        num_acc_grad_stages = len(layers)
        assert num_acc_grad_stages % 2 == 0

        stage_to_mesh = {
            i:
            (i if i < num_acc_grad_stages / 2 else num_acc_grad_stages - i - 1)
            for i, _ in enumerate(layers)
        }
        num_meshes = num_acc_grad_stages // 2
        stages = layers
        if submesh_shapes != None:
            assert all(shape == submesh_shapes[0] for shape in submesh_shapes)
        sliced_meshes = uniform_slice_mesh(mesh,
                                           num_meshes,
                                           submesh_shapes=submesh_shapes)
    else:
        raise ValueError(f"Unknown pipeline_stage_mode {pipeline_stage_mode}")

    # Check logical mesh shapes or assign default logical mesh shapes
    if logical_mesh_shapes is not None:
        assert len(logical_mesh_shapes) == len(sliced_meshes)
        for logical_mesh_shape, submesh in zip(logical_mesh_shapes,
                                               sliced_meshes):
            assert np.prod(logical_mesh_shape) == submesh.total_devices
    else:
        logical_mesh_shapes = [
            submesh.get_default_logical_mesh().shape
            for submesh in sliced_meshes
        ]
    if autosharding_global_configs is not None:
        assert len(autosharding_global_configs) == len(sliced_meshes)
    else:
        autosharding_global_configs = [{}] * len(sliced_meshes)

    for name in [
            "stage-construction", "stage-construction-dp",
            "stage-construction-compilation", "stage-construction-profiling"
    ]:
        if name in timers.timers:
            timers(name).stop()
    return (stages, stage_to_mesh, sliced_meshes, logical_mesh_shapes,
            autosharding_global_configs)


def get_stage_outvars(layers: Sequence[JaxPipelineComputation],
                      layer_assignment, global_outvars):
    """
    Perform liveness analysis to get the outvars of a stage that is used by
    another stage.
    Args:
        layers: clustered layers
        layer_assignment: the assignment of layers to stages
        global_outvars: global outvars

    Returns:
        A list of outvars for each stage
    """
    n_stages = len(layer_assignment)
    used = OrderedSet(global_outvars)
    stage_outvars = [OrderedSet() for _ in range(n_stages)]
    for stage_id, layer_ids in reversed(list(enumerate(layer_assignment))):
        for layer_id in layer_ids:
            for var in layers[layer_id].outvars:
                if var in used:
                    stage_outvars[stage_id].add(var)
            for var in layers[layer_id].invars:
                used.add(var)
    return stage_outvars
