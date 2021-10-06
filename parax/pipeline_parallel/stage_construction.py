import jax
import jax.numpy as jnp
import ray
import numba
import numpy as np
from typing import Sequence, Set, Tuple
from parax.pipeline_parallel.stage import JaxPipelineStage
from parax.device_mesh import VirtualMesh


@numba.jit(nopython=True)
def dp_impl(num_layers, num_devices, num_microbatches, submesh_choices, compute_cost, communication_cost, max_stage_cost):
    # For f, layer ID start from 1
    f = np.full((num_layers + 1, num_devices + 1), np.inf, dtype=np.float)
    f_stage_max = np.full((num_layers + 1, num_devices + 1), 0.0, dtype=np.float)
    f_argmin = np.full((num_layers + 1, num_devices + 1), -1, dtype=np.int)
    f[0] = 0
    for i in range(1, num_layers + 1):
        for j in range(1, num_devices + 1):
            for k in range(1, i + 1):
                for m, submesh in enumerate(submesh_choices):
                    s = np.prod(submesh)
                    if s <= j:
                        stage_cost = compute_cost[k, i, m] + communication_cost[k, m]
                        new_cost = f[k - 1, j - s] + stage_cost
                        if stage_cost <= max_stage_cost and new_cost < f[i, j]:
                            f[i, j] = new_cost
                            f_stage_max[i, j] = max(f_stage_max[i, j], f[i, j])
                            f_argmin[i, j] = (k, m)

    if np.isinf(f[num_layers, num_devices]):
        return np.inf, None

    total_cost = f[num_layers, num_devices] + (num_microbatches - 1) * f_stage_max[num_layers, num_devices]
    current_layer = num_layers
    current_devices = num_devices

    res = []
    while current_layer > 0 and current_devices > 0:
        start_layer, submesh_choice = f_argmin[current_layer, current_devices]
        res.append((start_layer - 1, current_layer), submesh_choice)
        current_layer = start_layer
        current_devices -= np.prod(submesh_choices[submesh_choice])

    return total_cost, res


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



def get_stage_and_mesh_assignments(layers: Sequence[JaxPipelineStage], mesh: VirtualMesh):
    num_layers = len(layers)
    submesh_choices = get_submesh_choices(mesh)
    submesh_sizes = [np.prod(submesh) for submesh in submesh_choices]

