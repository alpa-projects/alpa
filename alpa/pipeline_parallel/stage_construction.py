"""
Core implementations for stage construction algorithms.
The algorithm groups layers into pipeline stages.
"""
from collections import namedtuple
from dataclasses import dataclass
import logging
from typing import Sequence, List, Tuple, Dict, Union, Optional

from jax.core import Var
import numpy as np

from alpa.device_mesh import VirtualPhysicalMesh
from alpa.global_env import global_config
from alpa.pipeline_parallel.computation import (
    JaxPipelineComputation, merge_marked_jaxprs_with_named_call)
from alpa.pipeline_parallel.stage_profiling import (get_compute_cost,
                                                    last_compute_cost_file_name)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.timer import timers
from alpa.util import OrderedSet, maybe_numba_jit

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class AutoStageOption:
    """Options of auto stage construction algorithm."""
    # The search space of the physical submesh shapes.
    # Possible choices: {"power_of_two", "small_power_of_two", "all"}.
    submesh_physical_shape_space: str = "power_of_two"
    # The search space of the logical mesh shapes.
    # Possible choices: {"same_as_physical", "data_parallel_only",
    #                    "single_node_model_parallel", "all", "manual"}.
    # If "manual", the user needs to specify the logical mesh shape.
    manually_specified_submeshes: Sequence[Tuple[int, int]] = None
    # The search space for the logical mesh shapes.
    # Possible choices: {"all", "single_node_model_parallel",
    #                    "same_as_physical", "data_parallel_only",
    #                    "model_parallel_only"}.
    submesh_logical_shape_space: str = "single_node_model_parallel"
    # Profile only individual layers or composition different layers.
    # Possible choices: {"individual", "composition"}.
    layer_profile_mode: str = "composition"
    # The tolerance of imbalance in the auto-stage construction.
    stage_imbalance_tolerance: float = np.inf
    # Use HLO cost model for computational cost or profile for the cost.
    use_hlo_cost_model: bool = False
    # The filename of profiling result database.
    profiling_database_filename: Optional[str] = None
    # The file name of the cached compute cost.
    cached_profile_result: Optional[str] = None


@dataclass
class ManualStageOption:
    """Options of manual stage assignment."""
    # Layer IDs of each forward stage.
    forward_stage_layer_ids: Sequence[Sequence[int]]
    # The physical shapes of submeshes of each stage.
    submesh_physical_shapes: Sequence[Sequence[int]]
    # The logical shapes of submeshes of each stage.
    submesh_logical_shapes: Sequence[Sequence[int]]
    # The auto-sharding options of each stage.
    submesh_autosharding_option_dicts: Sequence[dict]


UniformStageOption = namedtuple("UniformStageOption", [])

StageOption = Union[AutoStageOption, ManualStageOption, UniformStageOption]

# Get results for debugging
last_forward_stage_layer_ids = None
last_submesh_shapes = None
last_logical_mesh_shapes = None
last_autosharding_option_dicts = None


def get_last_dp_result():
    """Gets the DP result of the last run."""
    return (last_compute_cost_file_name, last_forward_stage_layer_ids,
            last_submesh_shapes, last_logical_mesh_shapes,
            last_autosharding_option_dicts)


@maybe_numba_jit
def get_optimal_submeshes(best_s, f_argmin, num_devices, num_layers,
                          submesh_n_devices):
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
        current_devices -= submesh_n_devices[submesh_choice]
    assert (current_s == 0 and current_layer == num_layers and
            current_devices == 0)

    return res


@maybe_numba_jit
def training_dp_impl_2(num_layers, num_devices, submesh_sizes,
                       valid_idxs_and_costs, max_n_succ_stages):
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
    for d in range(1, num_devices + 1):
        for l, i, submesh_id, n_config, stage_cost in valid_idxs_and_costs:
            l, i, submesh_id, n_config = map(int, (l, i, submesh_id, n_config))
            n_submesh_devices = submesh_sizes[submesh_id]
            if n_submesh_devices <= d:
                for s in range(1, num_layers + 1):
                    if s - 1 > max_n_succ_stages[l, i, submesh_id, n_config]:
                        continue

                    new_cost = f[s - 1, i + 1,
                                 d - n_submesh_devices] + stage_cost
                    if new_cost < f[s, l, d]:
                        f[s, l, d] = new_cost
                        f_argmin[s, l, d] = (i + 1, submesh_id, n_config)
                        f_stage_max[s, l, d] = max(
                            f_stage_max[s - 1, i + 1, d - n_submesh_devices],
                            stage_cost)

    return f, f_stage_max, f_argmin


def training_dp_2(
    num_devices,
    num_microbatches,
    submesh_choices,
    compute_cost,
    max_n_succ_stages,
):
    """Faster implementation of the training DP algorihtm."""
    # TODO(zhuohan): Further verify the correctness of this implementation.
    timers("stage-construction-dp").start()

    num_layers = len(compute_cost)
    all_possible_stage_costs = np.sort(np.unique(compute_cost))
    best_cost = np.inf
    best_solution = None
    last_max_stage_cost = 0.0
    # FIXME(zhuohan): Set this gap as a tunable parameter in global config
    gap = 1e-6
    assert len(
        all_possible_stage_costs), "no solution in auto stage construction."

    submesh_sizes = np.array([n * m for (n, m) in submesh_choices],
                             dtype=np.int64)

    for max_stage_cost in all_possible_stage_costs:
        if max_stage_cost - last_max_stage_cost < gap:
            continue
        if max_stage_cost * num_microbatches >= best_cost:
            break

        # Lifts check for stage_cost <= t_max_stage_cost out of the inner dp
        # loop.
        valid_cost_idxs = np.transpose(
            (compute_cost <= max_stage_cost).nonzero())
        # This corresponds to the i of k <= i <= K from eqn. 3 in the alpa
        # paper.
        valid_cost_idxs = valid_cost_idxs[
            valid_cost_idxs[:, 0] <= valid_cost_idxs[:, 1]]
        if len(valid_cost_idxs) == 0:
            continue
        valid_costs = compute_cost[tuple(valid_cost_idxs.T)]
        valid_idxs_and_costs = np.hstack(
            [valid_cost_idxs, valid_costs[:, np.newaxis]])
        # Sort by descending layer idx because DP initializes
        # F[0, num_layers, 0] = 0
        valid_idxs_and_costs = valid_idxs_and_costs[np.flip(
            valid_cost_idxs[:, 1].argsort())]

        # Don't perform backtracking each time (do it only for the best
        # solution).
        f, f_stage_max, f_argmin = training_dp_impl_2(
            num_layers,
            num_devices,
            submesh_sizes,
            valid_idxs_and_costs,
            max_n_succ_stages,
        )

        best_s = f[:, 0, num_devices].argmin()
        best_total_cost = f[best_s, 0, num_devices]
        if np.isinf(best_total_cost):
            continue
        stage_cost = (num_microbatches - 1) * f_stage_max[best_s, 0,
                                                          num_devices]

        if best_total_cost + stage_cost < best_cost:
            best_cost = best_total_cost + stage_cost
            best_solution = best_s, f_argmin
        last_max_stage_cost = max_stage_cost

    assert best_solution is not None, (
        "Unable to find any solution to inter-op dp.")
    best_s, f_argmin = best_solution
    best_solution = get_optimal_submeshes(best_s, f_argmin, num_devices,
                                          num_layers, submesh_sizes)

    timers("stage-construction-dp").stop()
    return best_cost, best_solution


@maybe_numba_jit
def training_dp_impl(num_layers, num_devices, num_microbatches, submesh_choices,
                     num_autosharding_configs, compute_cost, max_n_succ_stages,
                     max_stage_cost):
    """The core implementation of the DP algorithm."""
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
    for s in range(1, num_layers + 1):  # pylint: disable=too-many-nested-blocks
        for i in range(num_layers - 1, -1, -1):
            for j in range(1, num_devices + 1):
                for k in range(num_layers, i, -1):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            # TODO(zhuohan): This level of for loop is not
                            #   necessary. It can be optimized by sorting
                            #   the logical mesh shapes.
                            for n_config in range(num_autosharding_configs):
                                if s - 1 <= max_n_succ_stages[i, k - 1, m,
                                                              n_config]:
                                    stage_cost = compute_cost[i, k - 1, m,
                                                              n_config]
                                    new_cost = f[s - 1, k, j -
                                                 n_submesh_devices] + stage_cost
                                    if (stage_cost <= max_stage_cost and
                                            new_cost < f[s, i, j]):
                                        f[s, i, j] = new_cost
                                        f_stage_max[s, i, j] = max(
                                            f_stage_max[s - 1, k,
                                                        j - n_submesh_devices],
                                            stage_cost)
                                        f_argmin[s, i, j] = (k, m, n_config)

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


def training_dp(num_layers, num_devices, num_microbatches, submesh_choices,
                num_autosharding_configs, compute_cost, max_n_succ_stages):
    """Auto stage dynamic programming."""
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
        cost, solution = training_dp_impl(num_layers, num_devices,
                                          num_microbatches, submesh_choices,
                                          num_autosharding_configs,
                                          compute_cost, max_n_succ_stages,
                                          max_stage_cost)
        if cost < best_cost:
            best_cost = cost
            best_solution = solution
        last_max_stage_cost = max_stage_cost

    timers("stage-construction-dp").stop()
    return best_cost, best_solution


@maybe_numba_jit
def inference_dp_impl(num_layers, num_devices, submesh_choices,
                      num_autosharding_configs, compute_cost):
    """The core implementation of the DP algorithm."""
    # For f, layer ID start from 0
    # f[#pipeline stages,
    #   layer id that is currently being considered,
    #   number of devices used]
    f = np.full((num_layers + 1, num_layers + 1, num_devices + 1),
                np.inf,
                dtype=np.float32)
    f_argmin = np.full((num_layers + 1, num_layers + 1, num_devices + 1, 3),
                       -1,
                       dtype=np.int32)
    f[0, 0, 0] = 0
    for s in range(1, num_layers + 1):  # pylint: disable=too-many-nested-blocks
        for i in range(1, num_layers + 1):
            for j in range(1, num_devices + 1):
                for k in range(0, i):
                    for m, submesh in enumerate(submesh_choices):
                        n_submesh_devices = np.prod(np.array(submesh))
                        if n_submesh_devices <= j:
                            for n_config in range(num_autosharding_configs):
                                stage_cost = compute_cost[k, i - 1, m, n_config]
                                new_cost = max(
                                    f[s - 1, k, j - n_submesh_devices],
                                    stage_cost)
                                if new_cost < f[s, i, j]:
                                    f[s, i, j] = new_cost
                                    f_argmin[s, i, j] = (k, m, n_config)

    best_s = -1
    best_total_cost = np.inf
    for s in range(1, num_layers + 1):
        if f[s, num_layers, num_devices] * s < best_total_cost:
            best_s = s
            best_total_cost = f[s, num_layers, num_devices] * s

    if np.isinf(best_total_cost):
        return np.inf, None

    current_s = best_s
    current_layer = num_layers
    current_devices = num_devices

    res = []
    while current_s > 0 and current_layer > 0 and current_devices > 0:
        next_end_layer, submesh_choice, autosharding_choice = (
            f_argmin[current_s, current_layer, current_devices])
        assert next_end_layer != -1
        res.append(((next_end_layer, current_layer), submesh_choice,
                    autosharding_choice))
        current_s -= 1
        current_layer = next_end_layer
        current_devices -= np.prod(np.array(submesh_choices[submesh_choice]))
    assert (current_s == 0 and current_layer == 0 and current_devices == 0)

    return best_total_cost, res


def inference_dp(num_layers, num_devices, submesh_choices,
                 num_autosharding_configs, compute_cost):
    """Auto stage dynamic programming."""
    timers("stage-construction-dp").start()
    cost, solution = inference_dp_impl(num_layers, num_devices, submesh_choices,
                                       num_autosharding_configs, compute_cost)
    solution = list(reversed(solution))
    timers("stage-construction-dp").stop()
    return cost, solution


def get_submesh_choices(
        num_hosts: int,
        num_devices_per_host: int,
        space: str,
        manually_specified_submeshes: Optional[Sequence[Tuple[int,
                                                              int]]] = None):
    """Gets the valid choices of submesh shapes."""
    if global_config.overwrite_submesh_choices is not None:
        return global_config.overwrite_submesh_choices
    submesh_choices = []

    # smaller submeshes:
    i = 1
    while i <= num_devices_per_host:
        submesh_choices.append((1, i))
        i *= 2
    assert submesh_choices[-1][1] == num_devices_per_host, (
        "Only supports the cases where num_devices_per_host is power of two, "
        f"while now num_devices_per_host = {num_devices_per_host}")

    # larger meshes:
    if space == "all":
        for i in range(2, num_hosts + 1):
            submesh_choices.append((i, num_devices_per_host))
    elif space == "power_of_two":
        i = 2
        while i <= num_hosts:
            submesh_choices.append((i, num_devices_per_host))
            i *= 2
    elif space == "small_power_of_two":
        i = 2
        while i <= min(num_hosts, 4):
            submesh_choices.append((i, num_devices_per_host))
            i *= 2
    elif space == "manual":
        submesh_choices = manually_specified_submeshes
    else:
        raise ValueError(f"Invalid submesh space: {space}")

    return tuple(submesh_choices)


def get_one_submesh_autosharding_config_choices(
        virtual_submesh: VirtualPhysicalMesh, space: str, batch_size: int):
    """
    Return a list of logical meshes and autosharding configs.
    Which will be used by the auto stage construction algorithm.

    Args:
        virtual_submesh: a submesh.
        space: The search space of the logical mesh shapes.
            possible choices: {"same_as_physical", "data_parallel_only",
                               "single_node_model_parallel", "all"}.
        batch_size: the batch size used.
    """
    results = []
    num_devices = virtual_submesh.num_devices
    if space in ["all", "single_node_model_parallel"]:
        if space == "all":
            max_mp_dimension = num_devices
        else:  # space == "single_node_model_parallel"
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
    elif space == "same_as_physical":
        results.append((virtual_submesh.get_logical_mesh(), {}))
    elif space == "data_parallel_only":
        results.append((virtual_submesh.get_logical_mesh((num_devices, 1)), {
            "force_batch_dim_to_mesh_dim": 0
        }))
    elif space == "model_parallel_only":
        results.append((virtual_submesh.get_logical_mesh((1, num_devices)), {
            "force_batch_dim_to_mesh_dim": 0
        }))
    else:
        raise ValueError(f"Invalid space for get_one_submesh_autosharding"
                         f"_config_choices: {space}")
    return results


def get_all_submesh_autosharding_config_choices(virtual_mesh, submesh_choices,
                                                space, batch_size):
    """Get all possible auto sharding config choices for all possible submesh
    shapes."""
    # A config is: Tuple(logical_mesh_shape, autosharding_option_dict).
    # Enumerate all (2D Mesh with force batch dim) + one (1D Mesh with mix batch
    # dim).
    autosharding_configs = []
    for submesh in submesh_choices:
        num_hosts, num_devices_per_host = submesh
        virtual_submesh = virtual_mesh.slice_2d(
            tuple(range(num_hosts)),
            (tuple(range(num_devices_per_host)),) * num_hosts)
        submesh_autosharding_configs = (
            get_one_submesh_autosharding_config_choices(virtual_submesh, space,
                                                        batch_size))
        autosharding_configs.append(submesh_autosharding_configs)

    # Pad all submesh to the maximum number of configs
    max_num_autosharding_configs = max(
        len(configs) for configs in autosharding_configs)
    for configs in autosharding_configs:
        configs += [None] * (max_num_autosharding_configs - len(configs))

    return autosharding_configs


def get_sliced_virtual_submeshes(virtual_mesh, submesh_shapes):
    """Slice the origin mesh into submeshes given submesh shapes."""
    num_hosts = virtual_mesh.num_hosts
    num_devices_per_host = virtual_mesh.num_devices_per_host
    submesh_sizes = [np.prod(submesh) for submesh in submesh_shapes]
    virtual_submeshes = [None] * len(submesh_shapes)
    assert sum(submesh_sizes) == virtual_mesh.num_devices
    sorted_submesh_indices = np.argsort(submesh_sizes, kind="stable")
    current_host_id = 0
    current_device_id = 0
    for i in reversed(sorted_submesh_indices):
        required_num_hosts, required_num_devices = submesh_shapes[i]
        if required_num_devices == num_devices_per_host:
            assert current_device_id == 0
            assert current_host_id + required_num_hosts <= num_hosts, (
                "Do not have enough hosts for the solution.")
            virtual_submeshes[i] = virtual_mesh.slice_2d(
                tuple(
                    range(current_host_id,
                          current_host_id + required_num_hosts)),
                (tuple(range(num_devices_per_host)),) * required_num_hosts)
            current_host_id += required_num_hosts
        else:
            assert required_num_hosts == 1
            assert required_num_devices < num_devices_per_host
            assert (current_device_id + required_num_devices <=
                    num_devices_per_host), (
                        "Do not have enough devices in a host for the solution")
            virtual_submeshes[i] = virtual_mesh.slice_2d([current_host_id], [
                tuple(
                    range(current_device_id,
                          current_device_id + required_num_devices))
            ])
            current_device_id += required_num_devices
            if current_device_id == num_devices_per_host:
                current_host_id += 1
                current_device_id = 0
    assert current_host_id == num_hosts
    assert current_device_id == 0
    return virtual_submeshes


def cluster_layers_and_slice_mesh(
        layers: Sequence[JaxPipelineComputation],
        virtual_mesh: VirtualPhysicalMesh, accumulator_mapping: Dict[Var, Var],
        acc_grad_invars: Sequence[Var], acc_grad_outvars: Sequence[Var],
        num_micro_batches: int, batch_size: int,
        jax_apply_layers: Sequence[JaxPipelineComputation],
        apply_grad_global_info: Tuple, pipeline_schedule: str,
        default_as_option: AutoShardingOption, stage_option: StageOption):
    """
    Stage-mesh assignment.

    This function clusters pipeline layers into stages, slice the device
    mesh into multiple submeshes, and assign the stages to the submeshes.
    We first profile the compute cost of layers on different choices
    of submeshes and find the optimal solution with DP.

    Args:
        layers: All the layers.
        virtual_mesh: The virtual device mesh.
        accumulator_mapping: The donation_mapping for the layers.
        acc_grad_invars: invars of the gradient accumulation layers.
        acc_grad_outvars: outvars of the gradient accumulation layers.
        num_micro_batches: The number of microbatches.
        batch_size: The micro batch size.
        jax_apply_layers: The apply gradient computations corresponding
          to each forward layers.
        pipeline_schedule: The pipeline schedule.
        default_as_option: The default auto-sharding option.
        stage_option: The options controling how to construct stages.
    """
    timers("stage-construction").start()

    inference_mode = (pipeline_schedule == "inference")
    if virtual_mesh.launched_physical_mesh_group is None:
        given_mesh = False
    else:
        given_mesh = True

    if inference_mode:
        num_layers = len(layers)
    else:
        # Assume each forward layer corresponds to a backward layer
        assert len(layers) % 2 == 0
        num_layers = len(layers) // 2

    if isinstance(stage_option, AutoStageOption):
        if given_mesh:
            # TODO(zhuohan): Implement the auto slicing with given mesh.
            raise NotImplementedError("automatically slicing layers with "
                                      "existing physical meshes is not"
                                      "supported yet.")

        submesh_choices = get_submesh_choices(
            virtual_mesh.num_hosts, virtual_mesh.num_devices_per_host,
            stage_option.submesh_physical_shape_space,
            stage_option.manually_specified_submeshes)
        autosharding_configs = get_all_submesh_autosharding_config_choices(
            virtual_mesh, submesh_choices,
            stage_option.submesh_logical_shape_space, batch_size)
        num_autosharding_configs = len(autosharding_configs[0])

        # Use DP to find the optimal solution.
        compute_cost, max_n_succ_stages = get_compute_cost(
            virtual_mesh, submesh_choices, autosharding_configs, layers,
            accumulator_mapping, acc_grad_invars, acc_grad_outvars,
            jax_apply_layers, apply_grad_global_info, num_micro_batches,
            default_as_option, stage_option, inference_mode)
        if inference_mode:
            _, solution = inference_dp(num_layers, virtual_mesh.num_devices,
                                       submesh_choices,
                                       num_autosharding_configs, compute_cost)
        else:
            _, solution = training_dp(num_layers, virtual_mesh.num_devices,
                                      num_micro_batches, submesh_choices,
                                      num_autosharding_configs, compute_cost,
                                      max_n_succ_stages)

        assert solution is not None, "no solution in auto stage construction."

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
        autosharding_option_dicts = [
            option_dict for _, option_dict in selected_autosharding_configs
        ]

        # Print and store the results
        print("Result forward_stage_layer_ids:", forward_stage_layer_ids)
        print("Result mesh_shapes:", submesh_shapes)
        print("Result logical_mesh_shapes:", logical_mesh_shapes)
        print("Result autosharding_option_dicts:", autosharding_option_dicts)
        global last_forward_stage_layer_ids, last_submesh_shapes
        global last_logical_mesh_shapes, last_autosharding_option_dicts
        last_forward_stage_layer_ids = forward_stage_layer_ids
        last_submesh_shapes = submesh_shapes
        last_logical_mesh_shapes = logical_mesh_shapes
        last_autosharding_option_dicts = autosharding_option_dicts
    elif isinstance(stage_option, ManualStageOption):
        # Check forward_stage_layer_ids is a partition of range(num_layers)
        forward_stage_layer_ids = stage_option.forward_stage_layer_ids
        last_layer_id = 0
        for stage_layer_ids in forward_stage_layer_ids:
            for layer_id in stage_layer_ids:
                assert layer_id == last_layer_id
                last_layer_id += 1
        assert last_layer_id == num_layers, (
            f"{last_layer_id} layers in stage option, but {num_layers} marked")
        submesh_shapes = stage_option.submesh_physical_shapes
        logical_mesh_shapes = (stage_option.submesh_logical_shapes or
                               submesh_shapes)
        autosharding_option_dicts = (
            stage_option.submesh_autosharding_option_dicts)
    elif isinstance(stage_option, UniformStageOption):
        if given_mesh:
            num_stages = num_layers
            submesh_shapes = [
                x.shape
                for x in virtual_mesh.launched_physical_mesh_group.meshes
            ]
            logical_mesh_shapes = submesh_shapes
        else:
            num_devices = virtual_mesh.num_devices
            num_stages = num_layers

            assert num_devices >= num_stages, "No enough devices"
            assert num_devices % num_stages == 0
            num_devices_per_mesh = num_devices // num_stages
            if num_devices_per_mesh > virtual_mesh.num_devices_per_host:
                assert (num_devices_per_mesh %
                        virtual_mesh.num_devices_per_host == 0)
                submesh_shape = (num_devices_per_mesh //
                                 virtual_mesh.num_devices_per_host,
                                 virtual_mesh.num_devices_per_host)
            else:
                assert (virtual_mesh.num_devices_per_host %
                        num_devices_per_mesh == 0)
                submesh_shape = (1, num_devices_per_mesh)
            submesh_shapes = [submesh_shape] * num_stages
            logical_mesh_shapes = [submesh_shape] * num_stages

        forward_stage_layer_ids = [[i] for i in range(num_layers)]
        autosharding_option_dicts = [{}] * num_stages
    else:
        raise ValueError(f"Invalid pipeline stage option: {stage_option}")

    if given_mesh:
        sliced_meshes = [
            mesh.get_virtual_physical_mesh()
            for mesh in virtual_mesh.launched_physical_mesh_group
        ]
    else:
        sliced_meshes = get_sliced_virtual_submeshes(virtual_mesh,
                                                     submesh_shapes)

    num_forward_stages = len(forward_stage_layer_ids)

    if inference_mode:
        stage_layer_ids = forward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages))
    else:
        backward_stage_layer_ids = [[
            2 * num_layers - 1 - i for i in reversed(layer_ids)
        ] for layer_ids in reversed(forward_stage_layer_ids)]
        stage_layer_ids = forward_stage_layer_ids + backward_stage_layer_ids
        stage_to_mesh = list(range(num_forward_stages)) + list(
            reversed(range(num_forward_stages)))

    stage_outvars = get_stage_outvars(layers, stage_layer_ids, acc_grad_outvars)
    merged_stages = []
    for stage_id, layer_ids in enumerate(stage_layer_ids):
        if len(layer_ids) == 1:
            merged_stages.append(layers[layer_ids[0]])
            continue

        stage_layer_jaxprs = [layers[i].closed_jaxpr() for i in layer_ids]
        stage_name = str(stage_id)
        merged_stage_jaxpr = merge_marked_jaxprs_with_named_call(
            stage_layer_jaxprs,
            stage_outvars[stage_id],
            accumulator_mapping,
            stage_name,
            wrap_with_marker=True)
        merged_stage = JaxPipelineComputation.from_closed_jaxpr(
            stage_name, merged_stage_jaxpr)
        merged_stages.append(merged_stage)
    stages = merged_stages

    # Check the validity of logical mesh shapes
    assert len(logical_mesh_shapes) == len(sliced_meshes)
    for logical_mesh_shape, submesh in zip(logical_mesh_shapes, sliced_meshes):
        assert np.prod(logical_mesh_shape) == submesh.num_devices

    if autosharding_option_dicts is not None:
        assert len(autosharding_option_dicts) == len(sliced_meshes)
    else:
        autosharding_option_dicts = [{}] * len(sliced_meshes)

    manual_stage_option = ManualStageOption(
        forward_stage_layer_ids, tuple(x.shape for x in sliced_meshes),
        logical_mesh_shapes, autosharding_option_dicts)

    timers("stage-construction").stop()
    return stages, stage_to_mesh, sliced_meshes, manual_stage_option


def get_stage_outvars(layers: Sequence[JaxPipelineComputation],
                      layer_assignment, global_outvars) -> List[OrderedSet]:
    """
    Get the outvars of a stage used by another stage by liveness analysis.

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
