from datetime import datetime
import math
from time import time
from typing import Sequence

import tqdm
import numba
import numpy as np
import ray

from parax.pipeline_parallel.computation import (JaxPipelineComputation,
                                                 merge_computation_jaxprs)
from parax.device_mesh import VirtualPhysicalMesh
from parax.pipeline_parallel.stage_profiling import (
    compute_apply_grad_invar_size, compute_intermediate_size,
    split_global_use_and_donate, generate_stage_info, compile_all,
    ProfileWorkerPool)
from parax.util import OrderedSet

GB = 1024 * 1024 * 1024
last_compute_cost_file_name = None
last_forward_stage_layer_ids = None
last_submesh_shapes = None


def get_last_dp_result():
    global last_compute_cost_file_name
    global last_forward_stage_layer_ids, last_submesh_shapes
    return (last_compute_cost_file_name, last_forward_stage_layer_ids,
            last_submesh_shapes)


@numba.jit(nopython=True)
def old_forward_dp_impl(num_layers, num_devices, num_microbatches,
                        submesh_choices, compute_cost, max_stage_cost):
    # For f, layer ID start from 1
    f = np.full((num_layers + 1, num_devices + 1), np.inf, dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_devices + 1),
                          0.0,
                          dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_devices + 1, 2), -1, dtype=np.int32)
    f[0, 0] = 0
    for i in range(1, num_layers + 1):
        for j in range(1, num_devices + 1):
            for k in range(1, i + 1):
                for m, submesh in enumerate(submesh_choices):
                    s = np.prod(np.array(submesh))
                    if s <= j:
                        stage_cost = compute_cost[k - 1, i - 1, m]
                        new_cost = f[k - 1, j - s] + stage_cost
                        if stage_cost <= max_stage_cost and new_cost < f[i, j]:
                            f[i, j] = new_cost
                            f_stage_max[i, j] = max(f_stage_max[k - 1, j - s],
                                                    stage_cost)
                            f_argmin[i, j] = (k, m)

    if np.isinf(f[num_layers, num_devices]):
        return np.inf, None

    total_cost = f[num_layers, num_devices] + (
        num_microbatches - 1) * f_stage_max[num_layers, num_devices]
    current_layer = num_layers
    current_devices = num_devices

    res = []
    while current_layer > 0 and current_devices > 0:
        start_layer, submesh_choice = f_argmin[current_layer, current_devices]
        assert start_layer != -1 and current_devices != -1
        res.append(((start_layer - 1, current_layer), submesh_choice))
        current_layer = start_layer - 1
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert current_layer == 0 and current_devices == 0

    return total_cost, res


@numba.jit(nopython=True)
def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
            compute_cost, max_n_succ_stages, max_stage_cost):
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
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 2),
                       -1,
                       dtype=np.int32)
    f[0, num_layers, 0] = 0
    for s in range(1, num_layers + 1):
        for i in range(num_layers - 1, -1, -1):
            for j in range(1, num_devices + 1):
                for k in range(num_layers, i, -1):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if (s - 1 <= max_n_succ_stages[i, k - 1, m] and
                                n_submesh_devices <= j):
                            stage_cost = compute_cost[i, k - 1, m]
                            new_cost = f[s - 1, k,
                                         j - n_submesh_devices] + stage_cost
                            if (stage_cost <= max_stage_cost and
                                    new_cost < f[s, i, j]):
                                f[s, i, j] = new_cost
                                f_stage_max[s, i, j] = max(
                                    f_stage_max[s - 1, k,
                                                j - n_submesh_devices],
                                    stage_cost)
                                f_argmin[s, i, j] = (k, m)

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
        next_start_layer, submesh_choice = f_argmin[current_s, current_layer,
                                                    current_devices]
        assert next_start_layer != -1 and current_devices != -1
        res.append(((current_layer, next_start_layer), submesh_choice))
        current_s -= 1
        current_layer = next_start_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == num_layers and
            current_devices == 0)

    return total_cost, res


def dp(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost,
       max_n_succ_stages):
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
                                 submesh_choices, compute_cost,
                                 max_n_succ_stages, max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
    assert best_solution is not None, "no solution in auto stage construction."
    return best_cost, best_solution


def get_submesh_choices(mesh: VirtualPhysicalMesh):
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
    for i in range(2, num_hosts + 1):
        submesh_choices.append((i, num_devices_per_host))

    return submesh_choices


def distributed_profile_on_mesh(meshes, layers, donation_mapping,
                                global_outvars, apply_grad_layers,
                                apply_grad_global_info):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    indices = list(range(2 * num_layers))
    compute_cost = np.full((num_layers, num_layers), np.inf)
    max_n_succ_stages = np.full((num_layers, num_layers), -1)
    stage_infos = []
    stage_indices = []
    stage_intermediate_vars = []
    profile_infos = []
    apply_grad_infos = []

    print("- Generate all stage infos (Jaxpr -> HLO)")
    # TODO(yonghao): only generate these info once for all mesh shapes
    for start in tqdm.tqdm(range(0, num_layers)):
        for end in tqdm.tqdm(range(start, num_layers), leave=False):
            layer_indices = (
                indices[start:end + 1] +
                indices[2 * num_layers - end - 1:2 * num_layers - start])
            selected_apply_grad_layers = [
                apply_grad_layers[idx] for idx in indices[start:end + 1]
            ]
            stage_name = "stage_{}_{}".format(start, end)
            (stage_info, intermediate_vars, profile_info,
             apply_info) = generate_stage_info(
                 layers,
                 layer_indices,
                 donation_mapping,
                 global_outvars,
                 stage_name,
                 insert_hook_after=end - start,
                 apply_grad_info=(selected_apply_grad_layers,
                                  *apply_grad_global_info))
            stage_infos.append(stage_info)
            stage_indices.append((start, end))
            stage_intermediate_vars.append(intermediate_vars)
            profile_infos.append(profile_info)
            apply_grad_infos.append(apply_info)

    print("- Compile all stages")
    # TODO(zhuohan): set the number of workers as a tunable parameter
    n_workers = int(max(ray.available_resources()["CPU"] // 2, 1))
    logical_mesh = meshes[0].get_default_logical_mesh()
    compiled_outputs = compile_all(stage_infos, logical_mesh, n_workers, 1)

    print("- Start all profiling tasks")
    profile_workers = ProfileWorkerPool(meshes)
    for (compiled_output, stage_info, intermediate_vars,
         apply_info) in zip(compiled_outputs, profile_infos,
                            stage_intermediate_vars, apply_grad_infos):
        (proto, config, in_shardings, out_shardings, hooked_proto,
         apply_in_shardings) = compiled_output
        intermediate_size = compute_intermediate_size(hooked_proto,
                                                      intermediate_vars,
                                                      config.logical_mesh_shape)
        apply_grad_input_size = compute_apply_grad_invar_size(
            apply_in_shardings, *apply_info, config.logical_mesh_shape)
        profile_workers.submit(lambda w, v: w.profile.remote(*v),
                               (compiled_output, stage_info, intermediate_size,
                                apply_grad_input_size))

    print("- Profile all stages")
    pbar = tqdm.tqdm(stage_indices)
    for start, end in pbar:
        cost, max_stage, debug_info = profile_workers.get_next()
        peak_memory, available_memory, intermediate_size, initial_size = debug_info
        compute_cost[start, end] = np.mean(cost)
        max_n_succ_stages[start, end] = max_stage
        pbar.write(f"cost[{start}, {end}]={compute_cost[start, end]},"
                   f" max_n_succ_stage={max_stage},"
                   f" Mem: peak={peak_memory / GB:.3f}GB,"
                   f" avail={available_memory / GB:.3f}GB,"
                   f" intermediate={intermediate_size / GB:.3f}GB,"
                   f" initial={initial_size / GB:.3f}GB")
    profile_workers.shutdown()
    return compute_cost, max_n_succ_stages


def get_compute_cost(virtual_mesh, submesh_choices, layers, donation_mapping,
                     global_outvars, apply_grad_layers, apply_grad_global_info):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    compute_cost = np.full((num_layers, num_layers, num_submesh_choices),
                           np.inf)
    max_n_succ_stages = np.full((num_layers, num_layers, num_submesh_choices),
                                -1)
    print("-" * 20 + " Automatic stage clustering " + "-" * 20)
    print(f"submesh_choices: {submesh_choices}")

    # Reverse submesh_choices to test larger meshes first
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        print(f"- Profiling for submesh {mesh_id} {submesh}:")
        num_hosts, num_devices = submesh
        tic = time()
        sliced_virtual_meshes = virtual_mesh.slice_profiling_submeshes(
            num_hosts, num_devices)
        mesh_compute_cost, mesh_max_n_succ_stages = distributed_profile_on_mesh(
            sliced_virtual_meshes, layers, donation_mapping, global_outvars,
            apply_grad_layers, apply_grad_global_info)

        compute_cost[:, :, mesh_id] = mesh_compute_cost
        max_n_succ_stages[:, :, mesh_id] = mesh_max_n_succ_stages
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
                                  jax_apply_layers=None,
                                  apply_grad_global_info=None,
                                  pipeline_stage_mode="uniform_layer_gpipe",
                                  cache_compute_cost=None,
                                  forward_stage_layer_ids=None,
                                  submesh_shapes=None):
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
    # For mesh-slicing's profiling, we can use the create_donation_mapping
    # to get a sketchy donation_mapping: only accumulate grad, no applygrad
    if pipeline_stage_mode in ["auto_gpipe", "manual_gpipe"]:
        # Assume each forward layer corresponds to a backward layer
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2
        submesh_choices = get_submesh_choices(mesh)
        if pipeline_stage_mode == "auto_gpipe":
            # use DP to find the optimal solution
            if cache_compute_cost is not None:
                # FIXME(zhuohan): load max_n_succ_stages
                compute_cost, max_n_succ_stages = np.load(cache_compute_cost,
                                                          allow_pickle=True)
            else:
                compute_cost, max_n_succ_stages = get_compute_cost(
                    mesh, submesh_choices, layers, donation_mapping,
                    global_outvars, jax_apply_layers, apply_grad_global_info)
            cost, solution = dp(num_layers, mesh.total_devices,
                                num_micro_batches, submesh_choices,
                                compute_cost, max_n_succ_stages)
            forward_stage_layer_ids = [
                list(range(start_id, end_id))
                for (start_id, end_id), _ in solution
            ]
            submesh_shapes = [submesh_choices[i] for _, i in solution]
            print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
            print("Result meshes:", submesh_shapes)
            global last_forward_stage_layer_ids, last_submesh_shapes
            last_forward_stage_layer_ids = forward_stage_layer_ids
            last_submesh_shapes = submesh_shapes
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
        # this mode resembles Megatron in terms of the uniformity of mesh shapes.
        num_acc_grad_stages = len(layers)
        stage_to_mesh = {
            i:
            (i if i < num_acc_grad_stages / 2 else num_acc_grad_stages - i - 1)
            for i, _ in enumerate(layers)
        }
        assert num_acc_grad_stages % 2 == 0
        num_meshes = num_acc_grad_stages // 2
        stages = layers
        if submesh_shapes != None:
            assert all(shape == submesh_shapes[0] for shape in submesh_shapes)
        sliced_meshes = uniform_slice_mesh(mesh,
                                           num_meshes,
                                           submesh_shapes=submesh_shapes)
    else:
        raise ValueError("Unknown pipeline_stage_mode", pipeline_stage_mode)
    return stages, stage_to_mesh, sliced_meshes


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
