import jax
import jax.numpy as jnp
import ray
import numba
import numpy as np
from time import time
from typing import Sequence, Set, Tuple
from parax.pipeline_parallel.stage import JaxPipelineStage
from parax.device_mesh import VirtualMesh
from parax.pipeline_parallel.mesh_slicing import (
    compile_and_profile_layer_cost_c, split_global_use_and_donate)


@numba.jit(nopython=True)
def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost, max_stage_cost):
    # For f, layer ID start from 1
    f = np.full((num_layers + 1, num_devices + 1), np.inf, dtype=np.float32)
    f_stage_max = np.full((num_layers + 1, num_devices + 1), 0.0, dtype=np.float32)
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
                            f_stage_max[i, j] = max(f_stage_max[k - 1, j - s], stage_cost)
                            f_argmin[i, j] = (k, m)

    if np.isinf(f[num_layers, num_devices]):
        return np.inf, None

    total_cost = f[num_layers, num_devices] + (num_microbatches - 1) * f_stage_max[num_layers, num_devices]
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


def dp(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost):
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    gap = 1e-6
    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost * num_microbatches >= best_cost:
            break
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        cost, solution = dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost, max_stage_cost)
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


def profile_on_mesh(layers, mesh, donation_mapping, global_outvars):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    all_invars = [set(layer.invars) for layer in layers]
    indices = list(range(2 * num_layers))
    compute_cost = np.full((num_layers, num_layers), np.inf)
    for start in range(0, num_layers):
        for end in range(start, num_layers):
            layer_collections = layers[start:end + 1] + layers[2 * num_layers - end - 1:2 * num_layers - start]
            layer_indices = indices[start:end + 1] + indices[2 * num_layers - end - 1:2 * num_layers - start]
            _, global_used_list = split_global_use_and_donate(layer_collections, layer_indices, all_invars, donation_mapping, global_outvars)
            donate_invars_list = [[False for _ in stage.invars] for stage in layer_collections]
            cost, in_specs, out_specs = compile_and_profile_layer_cost_c(layer_collections, mesh, donate_invars_list, global_used_list)
            compute_cost[start, end] = np.mean(cost)
    return compute_cost


def get_compute_cost(layers, submesh_choices, virtual_mesh, donation_mapping, global_outvars):
    assert len(layers) % 2 == 0
    num_layers = len(layers) // 2
    num_submesh_choices = len(submesh_choices)
    compute_cost = np.full((num_layers, num_layers, num_submesh_choices), np.inf)
    for mesh_id, submesh in enumerate(submesh_choices):
        num_hosts, num_devices = submesh
        sliced_virtual_mesh = virtual_mesh.slice_2d(list(range(num_hosts)), [list(range(num_devices)) for _ in range(num_hosts)])
        mesh = sliced_virtual_mesh.get_physical_mesh()
        tic = time()
        mesh_compute_cost = profile_on_mesh(layers, mesh, donation_mapping, global_outvars)
        compute_cost[:, :, mesh_id] = mesh_compute_cost
        toc = time()
        mesh.shutdown()
        print(f'profiling for submesh {mesh_id} {submesh} takes {toc - tic} seconds')
        print(f'profiled costs are: {compute_cost[:, :, mesh_id]}')
        print('=' * 30)

def get_mesh_slicing_scheme(virtual_mesh, submesh_choices, solution):
    submeshes = [submesh_choices[choice] for _, choice in solution]
    submesh_sizes = [np.prod(submesh) for submesh in submeshes]
    assert sum(submesh_sizes) == virtual_mesh
    # TODO(zhuohan): Finish here


def get_stage_and_mesh_assignments(layers: Sequence[JaxPipelineStage], mesh: VirtualMesh):
    num_layers = len(layers)
    submesh_choices = get_submesh_choices(mesh)
    submesh_sizes = [np.prod(submesh) for submesh in submesh_choices]

