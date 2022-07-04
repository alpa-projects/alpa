"""
The data strcutures to save all configurations/strategies of
a parallel execution plan.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from jax.core import ShapedArray
from jax.interpreters import pxla


@dataclass
class PlacementSpec:
    """Specify how a tensor is stored distributedly."""
    aval: ShapedArray
    mesh_ids: Sequence[int]
    sharding_specs: Sequence[pxla.ShardingSpec]


@dataclass
class StagePlan:
    """The parallel plan for a single sharded stage."""
    build_random_seed: int
    logical_mesh_shape: Tuple[int]
    all_gather_threshold: int
    all_reduce_threshold: int
    auto_sharding_option: "AutoShardingOption"
    auto_sharding_solution_vector: np.ndarray
    auto_sharding_objective: int


@dataclass
class PipelinePlan:
    """The parallel plan for a pipeline."""
    pipeline_schedule: str
    layer_option: "LayerOption"
    manual_stage_option: "ManualStageOption"


@dataclass
class ClusterInfo:
    num_hosts: int
    num_devices_per_host: int


@dataclass
class ParallelPlan:
    """The global parallel plan."""
    cluster_info: ClusterInfo
    num_micro_batches: int
    auto_sharding_option: "AutoShardingOption"
    pipeline_plan: PipelinePlan
    input_placement_specs: Sequence[PlacementSpec]


def plan_to_method(plan: ParallelPlan) -> "ParallelMethod":
    """Convert a parallel plan to a parallel method."""
    # pylint: disable=import-outside-toplevel
    from alpa.parallel_method import ShardParallel, PipeshardParallel

    if plan.pipeline_plan is None:
        return ShardParallel(num_micro_batches=plan.num_micro_batches,
                             auto_sharding_option=plan.auto_sharding_option)
    else:
        return PipeshardParallel(
            num_micro_batches=plan.num_micro_batches,
            default_auto_sharding_option=plan.auto_sharding_option,
            pipeline_schedule=plan.pipeline_plan.pipeline_schedule,
            layer_option=plan.pipeline_plan.layer_option,
            stage_option=plan.pipeline_plan.manual_stage_option)
