import jax
import jax.numpy as jnp
import ray
import numba
import numpy as np
from datetime import datetime
from time import time
from typing import Sequence, Set, Tuple
from parax.pipeline_parallel.computation import JaxPipelineComputation
from parax.device_mesh import VirtualMesh
from parax.pipeline_parallel.stage_profiling import (
    compile_and_profile_stage_compute_cost, split_global_use_and_donate)


@numba.jit(nopython=True)
def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
            compute_cost, max_stage_cost):
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


def dp(num_layers, num_devices, num_microbatches, submesh_choices,
       compute_cost):
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl(num_layers, num_devices, num_microbatches,
                                 submesh_choices, compute_cost, max_stage_cost)
        if solution is not None:
            solution = list(reversed(solution))
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
    return best_cost, best_solution


def get_submesh_choices(mesh: VirtualMesh):
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


def profile_on_mesh(mesh, layers, donation_mapping, global_outvars):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    indices = list(range(2 * num_layers))
    compute_cost = np.full((num_layers, num_layers), np.inf)
    for start in range(0, num_layers):
        for end in range(start, num_layers):
            layer_indices = indices[start:end +
                                    1] + indices[2 * num_layers - end -
                                                 1:2 * num_layers - start]
            local_donation_mapping, global_used_list, selected_layers = (
                split_global_use_and_donate(layers, layer_indices,
                                            donation_mapping, global_outvars))
            cost, in_specs, out_specs = compile_and_profile_stage_compute_cost(
                selected_layers, mesh, local_donation_mapping, global_used_list)
            compute_cost[start, end] = np.mean(cost)
    return compute_cost


def get_compute_cost(virtual_mesh, submesh_choices, layers, donation_mapping,
                     global_outvars):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    compute_cost = np.full((num_layers, num_layers, num_submesh_choices),
                           np.inf)
    # Reverse submesh_choices to test larger meshes first
    for mesh_id, submesh in reversed(list(enumerate(submesh_choices))):
        num_hosts, num_devices = submesh
        sliced_virtual_mesh = virtual_mesh.slice_2d(
            list(range(num_hosts)),
            [list(range(num_devices)) for _ in range(num_hosts)])
        mesh = sliced_virtual_mesh.get_physical_mesh()
        tic = time()
        mesh_compute_cost = profile_on_mesh(mesh, layers, donation_mapping,
                                            global_outvars)
        compute_cost[:, :, mesh_id] = mesh_compute_cost
        toc = time()
        mesh.shutdown()
        print(
            f'profiling for submesh {mesh_id} {submesh} takes {toc - tic} seconds'
        )
        print(f'profiled costs are: {mesh_compute_cost}')
        print('=' * 30)
    return compute_cost


def get_sliced_virtual_submeshes(virtual_mesh, submesh_choices, solution):
    num_hosts = virtual_mesh.num_hosts
    num_devices_per_host = virtual_mesh.num_devices_per_host
    submeshes = [submesh_choices[choice] for _, choice in solution]
    submesh_sizes = [np.prod(submesh) for submesh in submeshes]
    virtual_submeshes = [None] * len(submeshes)
    assert sum(submesh_sizes) == virtual_mesh.total_devices
    sorted_submesh_indices = np.argsort(submesh_sizes)
    current_host_id = 0
    current_device_id = 0
    for i in reversed(sorted_submesh_indices):
        required_num_hosts, required_num_devices = submeshes[i]
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


def get_stage_and_mesh_assignments(mesh: VirtualMesh,
                                   layers: Sequence[JaxPipelineComputation],
                                   donation_mapping,
                                   global_outvars,
                                   num_microbatches,
                                   compute_cost=None):
    """
    Automatically cluster layers into stages, slice the device mesh
    into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        mesh (VirtualMesh): The cluser device mesh.
        layers (Sequence[JaxPipelineComputation]): All the layers.
        donation_mapping: The donation_mapping for the layers.
        global_outvars: Global outvars of the layers.
        num_microbatches: Number of microbatches for GPipe
        compute_cost (Optional): Override the profiling results.

    Returns:
        stage_layer_ids (List[List[int]]): The layer IDs of each stage.
        sliced_meshes (List[VirtualMesh]): The shapes of all submeshes.
    """
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    submesh_choices = get_submesh_choices(mesh)
    if compute_cost is None:
        compute_cost = get_compute_cost(mesh, submesh_choices, layers,
                                        donation_mapping, global_outvars)
        np.save(
            f"compute-cost-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.npz",
            compute_cost)
    cost, solution = dp(num_layers, mesh.total_devices, num_microbatches,
                        submesh_choices, compute_cost)
    stage_layer_ids = [
        list(range(start_id, end_id)) for (start_id, end_id), _ in solution
    ]
    print("Result stage_layer_ids:", stage_layer_ids)
    print("Result meshes:", [submesh_choices[i] for _, i in solution])
    sliced_meshes = get_sliced_virtual_submeshes(mesh, submesh_choices,
                                                 solution)
    return stage_layer_ids, sliced_meshes


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
    used = set(global_outvars)
    stage_outvars = [set() for _ in range(n_stages)]
    for stage_id, layer_ids in reversed(list(enumerate(layer_assignment))):
        for layer_id in layer_ids:
            for var in layers[layer_id].outvars:
                if var in used:
                    stage_outvars[stage_id].add(var)
            for var in layers[layer_id].invars:
                used.add(var)
    return stage_outvars
