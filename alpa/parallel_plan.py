"""
The data strcutures to save all configurations/strategies of
a parallel execution plan.
"""
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np
from jax.interpreters import pxla
from jax.tree_util import PyTreeDef


@dataclass
class PlacementSpec:
    """Specify how a tensor is stored distributedly."""
    mesh_ids: Sequence[int]
    sharding_specs: Sequence[pxla.ShardingSpec]


@dataclass
class StagePlan:
    """The parallel plan for a single sharded stage."""
    build_random_seed: int
    logical_mesh_shape: Tuple[int]
    all_gather_threshold: int
    all_reduce_threshold: int
    auto_sharding_solution_vector: np.ndarray
    auto_sharding_objective: int


@dataclass
class PipelinePlan:
    """The parallel plan for a pipeline."""
    forward_stage_layer_ids: Sequence[Sequence[int]]
    submesh_physical_shapes: Sequence[Sequence[int]]
    submesh_logical_shapes: Sequence[Sequence[int]]
    submesh_autosharding_option_dicts: Sequence[dict]


@dataclass
class ParallelPlan:
    """The global parallel plan."""
    pipeline_plan: PipelinePlan
    stage_plans: Sequence[StagePlan]
    input_placement: PyTreeDef
    version: str
