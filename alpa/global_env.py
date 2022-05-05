"""All global configurations for this project."""
import copy
import os
from typing import Optional, Sequence, Tuple

import numpy as np


class AutoShardingOption:
    """Options of the auto-sharding solver."""

    def __init__(self):
        self.allow_all_gather = True  # Whether to allow all-gather during re-sharding.
        self.allow_all_to_all = True  # Whether to allow all-to-all during re-sharding.
        self.allow_replicated_parameters = True  # Whether to allow replicated parameters.
        self.force_data_parallel = False  # Whether to forcibly generate data-parallel.
        self.force_batch_dim_to_mesh_dim = None  # Forcibly map the batch dimension to
        # a mesh dimension.
        self.force_zero_stage_3 = False  # Whether to forcibly generate a strategy similar to
        # ZeRO optimizer stage 3.
        self.force_zero_stage_3_all_gather_threshold = 1 << 25  # The threshold of all-gather combiner
        # if force_zero_stage_3 is true.
        self.prefer_reduce_scatter = False  # Prefer reduce-scatter over all-reduce.
        self.allow_mixed_mesh_shape = False  # Allow mixed 1d mesh and 2d mesh shape.
        self.allow_recompute_heavy_op = False  # Allow replicated dot computation.
        self.force_simple_heuristic = ""  # If it is not empty, forcibly use a simple heuristic
        # instead of the ILP solver.
        self.all_reduce_threshold = 1 << 60  # The threshold of all-reduce combiner in bytes.

    def deepcopy_and_update(self, new_values: dict):
        """Make a deepcopy and update some keys with new values."""
        ret = copy.copy(self)
        for k, v in new_values.items():
            assert hasattr(ret, k)
            setattr(ret, k, v)
        return ret

    def backup(self):
        """Backup the configs."""
        return copy.deepcopy(self.__dict__)

    def restore(self, saved_dict):
        """Restore the configs from a backup."""
        self.__dict__ = saved_dict


class GlobalConfig:
    """The global configuration of alpa.

    See also the docstring of `set_parallelize_options` for the meanings
    of the member variables.
    """

    def __init__(self):
        ########## Options of both shard_parallel and pipeline_parallel ##########
        self.devices = None
        self.strategy = "shard_parallel"
        self.memory_budget_per_device = None
        self.num_micro_batches = None  # If is not None, gradient accumulation will
        # be enable.
        self.default_autosharding_option = AutoShardingOption()

        ########## Options of device mesh ##########
        self.xla_client_mem_fraction = float(
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.9))
        self.xla_gpu_autotune_level = 4
        self.delete_remote_buffers_threshold = 500
        # use AWS EFA network interface
        self.use_aws_efa = os.environ.get("ALPA_USE_AWS_EFA", "").lower() in [
            "true", "1"
        ]

        ########## Options of shard_parallel ##########
        self.shard_parallel_search_logical_mesh_shape = False
        self.shard_parallel_mesh_shape_search_mode = "cost_model"
        self.shard_parallel_mesh_shape_search_log_file = None
        self.shard_parallel_sync_for_timer = False

        ########## Options of pipeline_parallel ##########
        self.pipeline_stage_mode = "uniform_stage"
        self.pipeline_parallel_schedule = "1f1b"

        # manual stage
        self.forward_stage_layer_ids = None
        self.sub_physical_mesh_shapes = None
        self.sub_logical_mesh_shapes = None
        self.submesh_autosharding_option_dicts = None

        # auto stage
        self.profile_with_whole_ray_cluster = True
        self.cache_compute_cost = None  # The path to the file containing the compute cost profile
        self.submesh_choices_mode = "power_of_two"
        self.logical_mesh_search_space = "default"
        self.auto_stage_construction_imbalance_tolerance = np.inf
        self.use_hlo_cost_model = False
        self.profiling_database_filename = None
        self.debug_with_local_runtime = False
        self.profile_timeout = 500
        self.profile_maximum_retry = 2
        self.overwrite_submesh_choices = None

        ########## Options of pipeline runtime ##########
        self.pipeline_runtime_mode = "paper"  # or "production"
        self.pipeline_distributed_compile = True  # Whether to use distributed compilation
        # in pipeline parallel for each stage. Disabling it helps debug.
        self.pipeline_use_signal_send_recv = False
        self.precompile_resharding_tasks = True
        self.use_scatter_gather = True
        self.eagerly_create_communicators = True
        self.use_memzero_for_gradient_accumulation = False

        ########## Options of XLA compilation ##########
        self.build_random_seed = 42
        self.remat_using_while = False

        ########## Options of benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False

        ########## Options of logging ##########
        self.print_xla_compilation_time = False

        ########## Options of ray namespace ##########
        self.default_ray_namespace_prefix = "alpa-train"
        self.unittest_ray_namespace_prefix = "alpa-unittest"

    def backup(self):
        """Backup the configs."""
        return copy.deepcopy(self.__dict__)

    def restore(self, saved_dict):
        """Restore the configs from a backup."""
        self.__dict__ = saved_dict

    def update_with_dict(self, value_dict):
        """Update the config with values from a dictionary."""
        for k, v in value_dict.items():
            assert hasattr(self, k), k
            setattr(self, k, v)


global_config = GlobalConfig()


def set_parallelize_options(
        devices=None,
        strategy: str = "shard_parallel",
        memory_budget_per_device: Optional[float] = None,
        num_micro_batches: Optional[int] = None,
        # shard-parallel
        search_logical_mesh_shape: bool = False,
        mesh_shape_search_mode: str = "cost_model",
        mesh_shape_search_log_file: Optional[str] = None,
        # pipeline-parallel
        pipeline_stage_mode: str = "uniform_stage",
        pipeline_parallel_schedule: str = "1f1b",
        forward_stage_layer_ids: Optional[Sequence[Sequence[int]]] = None,
        sub_physical_mesh_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        sub_logical_mesh_shapes: Optional[Sequence[Tuple[int, int]]] = None,
        submesh_autosharding_option_dicts: Optional[Sequence[dict]] = None,
        cache_compute_cost: Optional[str] = None,
        logical_mesh_search_space: str = "default",
        auto_stage_construction_imbalance_tolerance: float = np.inf,
        use_hlo_cost_model: bool = False,
        profiling_database_filename: Optional[str] = None):
    """
    Set the global options for all @parallelize decorator.

    Args:
      devices: The device cluster.
      strategy: The parallelization strategy.
        Possible choices: {"shard_parallel", "pipeshard_parallel", "local_pipeline_parallel"}.
      memory_budget_per_device: The memory budget of one device in bytes.
      num_micro_batches: The number of micro-batches for gradient accumulation.
      search_logical_mesh_shape: Only used for shard_parallel.
        Whether to include the choices of logical mesh shape into the search space.
      mesh_shape_search_mode: Only used for shard_parallel.
        Whether to use cost model or real measurement to pick the logical mesh shape.
        Possible choices: {"cost_model", "measurement"}.
      mesh_shape_search_log_file: Only used for shard_parallel.
        The file to store measurement records of logical mesh shape search.
      pipeline_stage_mode: Only used for pipeshard_parallel.
        The algorithm used to construct pipeline stages.
        Possible choices: {"auto_stage", "manual_stage", "uniform_stage"}
      pipeline_parallel_schedule: The pipeline schedule. Possible choices: {"gpipe", "1f1b"}.
      forward_stage_layer_ids: Only used for pipeshard_parallel with manual_stage
        Layer IDs of each forward stage.
      sub_physical_mesh_shapes: Only used for pipeshard_parallel with manual_stage
        The shapes of submeshes for each forward stage.
      sub_logical_mesh_shapes: Only used for pipeshard_parallel with manual_stage
        The logical shapes of submeshes for each stage.
      submesh_autosharding_option_dicts: Only used for pipeshard_parallel with manual_stage.
        The auto-sharding options of each stage.
      cache_compute_cost: Only used for pipeshard_parallel with auto_stage
        The file name of the cached compute cost.
      logical_mesh_search_space: Only used for pipeshard_parallel with auto_stage
        The search space for the logical mesh shape.
        Possible choices: {"default", "all", "single_node_model_parallel"}.
      auto_stage_construction_imbalance_tolerance: Only used for pipeshard_parallel with auto_stage
        The tolerance of imbalance in the auto-stage construction.
      use_hlo_cost_model: Only used for pipeshard_parallel with auto_stage.
        Whether to use the Hlo instruction cost model for pipeline profiling.
      profiling_database_filename: Only used for pipeshard_parallel with auto_stage.
        The filename of profiling result database.
    """
    global global_config  # pylint: disable=global-variable-not-assigned

    global_config.devices = devices
    global_config.strategy = strategy
    global_config.memory_budget_per_device = memory_budget_per_device
    global_config.num_micro_batches = num_micro_batches

    # shard-parallel
    global_config.shard_parallel_search_logical_mesh_shape = search_logical_mesh_shape
    global_config.shard_parallel_mesh_shape_search_mode = mesh_shape_search_mode
    global_config.shard_parallel_mesh_shape_search_log_file = mesh_shape_search_log_file

    # pipeline-parallel
    global_config.pipeline_stage_mode = pipeline_stage_mode
    global_config.pipeline_parallel_schedule = pipeline_parallel_schedule
    global_config.forward_stage_layer_ids = forward_stage_layer_ids
    global_config.sub_physical_mesh_shapes = sub_physical_mesh_shapes
    global_config.sub_logical_mesh_shapes = sub_logical_mesh_shapes
    global_config.submesh_autosharding_option_dicts = submesh_autosharding_option_dicts
    global_config.cache_compute_cost = cache_compute_cost
    global_config.logical_mesh_search_space = logical_mesh_search_space
    global_config.auto_stage_construction_imbalance_tolerance = auto_stage_construction_imbalance_tolerance
    global_config.use_hlo_cost_model = use_hlo_cost_model
    global_config.profiling_database_filename = profiling_database_filename


is_worker = os.environ.get("ALPA_IS_WORKER", "False") == "True"

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=false"
