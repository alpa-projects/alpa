"""All global configurations for this project."""
import os
import copy

import numpy as np


class GlobalConfig:
    """The global configuration of parax.

    See also the docstring of `set_parallelize_options` for the meanings
    of the member variables.
    """

    def __init__(self):
        ########## Options for @parallelize decorator ##########
        self.devices = None
        self.strategy = "shard_parallel"
        self.memory_budget_per_device = None
        self.num_micro_batches = None  # If is not None, gradient accumulation will
        # be enable.

        ########## Options of shard parallel ##########
        self.search_logical_mesh_shape = False
        self.mesh_shape_search_mode = "cost_model"
        self.mesh_shape_search_log_file = None
        self.profile_communication = False

        self.cache_folder = "parax_cache"
        self.cache_auto_sharding_ilp_solution = False

        ########## Options of pipeline parallel ##########
        self.pipeline_stage_mode = "uniform_layer_gpipe"
        self.profile_with_whole_ray_cluster = True
        self.cache_compute_cost = None  # The path to the file containing the compute cost profile
        self.forward_stage_layer_ids = None
        self.sub_physical_mesh_shapes = None
        self.sub_logical_mesh_shapes = None
        self.submesh_autosharding_global_configs = None
        self.submesh_choices_mode = "power_of_two"
        self.logical_mesh_search_space = "default"
        self.auto_stage_construction_imbalance_tolerance = np.inf
        self.pipeline_parallel_schedule = "1f1b"
        self.pipeline_runtime_mode = "paper"  # or "production"
        self.use_hlo_cost_model = False
        self.profiling_database_filename = None
        self.with_physical_mesh = True
        self.profile_timeout = 600
        self.profile_maximum_retry = 2

        ########## Options of auto-sharding solver ##########
        self.allow_all_gather = True  # Wether allow all-gather during re-sharding.
        self.allow_all_to_all = True  # Wether allow all-to-all during re-sharding.
        self.allow_replicated_parameters = True  # Whether allow replicated parameters.
        self.force_data_parallel = False  # Whether force to generate data-parallel.
        self.force_batch_dim_to_mesh_dim = None  # Forcibly map the batch dimension to
        # a mesh dimension.
        self.force_zero_stage_3 = False  # Whether force to generate a strategy similar to
        # ZeRO optimizer stage 3.
        self.force_zero_stage_3_all_gather_threshold = 1 << 25  # The threshold of all-gather combiner
        # if force_zero_stage_3 is true.
        self.prefer_reduce_scatter = False  # Prefer reduce-scatter over allreduce.
        self.allow_mixed_mesh_shape = False  # Allow mixed 1d mesh and 2d mesh shape.
        self.allow_recompute_heavy_op = False  # Allow replicated dot computation.
        self.force_simple_heuristic = ""  # If it is not empty, forcibly use a simple heuristic
        # instead of the ILP solver.

        ########## Options of pipeline runtime ##########
        self.pipeline_distributed_compile = True  # Whether to use distributed compilation
        # in pipeline parallel for each stage. Disabling it helps debug.
        self.pipeline_use_signal_send_recv = False
        self.precompile_resharding_tasks = True
        self.use_scatter_gather = True

        ########## Options of XLA compilation ##########
        self.build_random_seed = 42
        self.remat_using_while = False

        ########## Options of benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False
        self.fix_physical_mesh_shape = None

        ########## Options of logging ##########
        self.print_xla_compilation_time = False

        ########## Options of ray namespace ##########
        self.default_ray_namespace_prefix = "parax-train"
        self.unittest_ray_namespace_prefix = "parax-unittest"

    def backup(self):
        """Backup the configs."""
        return copy.copy(self.__dict__)

    def restore(self, saved_dict):
        """Restore the configs from a backup."""
        global_config.__dict__ = saved_dict


global_config = GlobalConfig()


def set_parallelize_options(
        devices=None,
        strategy: str = "shard_parallel",
        memory_budget_per_device=None,
        num_micro_batches=None,
        # shard-parallel
        search_logical_mesh_shape=False,
        mesh_shape_search_mode="cost_model",
        mesh_shape_search_log_file=None,
        profile_communication=False,
        cache_folder="parax_cache",
        cache_auto_sharding_ilp_solution=False,
        # pipeline-parallel
        pipeline_stage_mode="uniform_layer_gpipe",
        cache_compute_cost=None,
        forward_stage_layer_ids=None,
        sub_physical_mesh_shapes=None,
        sub_logical_mesh_shapes=None,
        submesh_autosharding_global_configs=None,
        logical_mesh_search_space="default",
        auto_stage_construction_imbalance_tolerance=np.inf,
        pipeline_parallel_schedule="1f1b",
        use_hlo_cost_model=False,
        profiling_database_filename=None):
    """
    Set the global options for all @parallelize decorator.

    Args:
      devices: The device cluster.
      strategy: The parallelization strategy.
        Possible choices: {"shard_parallel", "pmap_data_parallel",
        "shard_data_parallel", "local_pipeline_parallel", "3d_parallel"}.
      memory_budget_per_device (Optional[float]): The memory budget of one
        device in bytes.
      search_logical_mesh_shape (bool): Whether to include the choices of
        logical mesh shape into the search space.
      mesh_shape_search_mode (str): Whether to use cost model or real
        measurement to pick the logical mesh shape. Possible choices: {
        "cost_model", "measurement"}.
      mesh_shape_search_log_file (Optional[str]): The file to store measurement
        records of logical mesh shape search.
      num_micro_batches (int): The number of micro batches in pipeline parallel
        and gradient accumulation.
      profile_communication (bool): Whether to use the profiled communication
        cost for the auto-sharding pass.
      cache_folder (str): The folder to store cached profiling results and
        strategies.
      cache_auto_sharding_ilp_solution (bool): Whether to cache the ilp
        solution generated during auto-sharding pass.
      pipeline_stage_mode (str): The algorithm used to construct pipeline
        stages. Possible choice: {"uniform_layer_gpipe", "auto_gpipe",
        "manual_gpipe"}
      cache_compute_cost (Optional[str]): The file name of the cached compute
        cost. Used for "auto_gpipe".
      forward_stage_layer_ids (Optional[List[List[int]]]): Layer IDs of each
        forward stage. Used for "manual_gpipe".
      sub_physical_mesh_shapes (Optional[List[Tuple[int, int]]]): The shapes of submeshes
        for each forward stage. Used for "manual_gpipe".
      sub_logical_mesh_shapes (Optional[List[Tuple[int, int]]]): the logical shapes of
        submeshes for each forward stage. Used for manual layer slicing.
      submesh_autosharding_global_configs (Optional[List[Dict]]): The global
        configuration for auto-sharding of submeshes. Used for manual layer
        slicing.
      logical_mesh_search_space (str): The search space for the logical mesh
        shape. Possible choices: {"default", "all", "single_node_model_parallel"}.
      auto_stage_construction_imbalance_tolerance (float): The tolerance of
        imbalance in the auto-stage construction.
      pipeline_parallel_schedule (str): the pipeline schedule, "gpipe" or "1f1b".
      use_hlo_cost_model (bool): Whether use the Hlo instruction cost model for pipeline profiling.
      profiling_database_filename (str): The filename of profiling result database.
    """
    global global_config

    global_config.devices = devices
    global_config.strategy = strategy
    global_config.memory_budget_per_device = memory_budget_per_device
    global_config.search_logical_mesh_shape = search_logical_mesh_shape
    global_config.mesh_shape_search_mode = mesh_shape_search_mode
    global_config.mesh_shape_search_log_file = mesh_shape_search_log_file
    global_config.num_micro_batches = num_micro_batches
    global_config.profile_communication = profile_communication
    global_config.cache_folder = cache_folder
    global_config.cache_auto_sharding_ilp_solution = cache_auto_sharding_ilp_solution
    global_config.pipeline_stage_mode = pipeline_stage_mode
    global_config.cache_compute_cost = cache_compute_cost
    global_config.forward_stage_layer_ids = forward_stage_layer_ids
    global_config.sub_physical_mesh_shapes = sub_physical_mesh_shapes
    global_config.sub_logical_mesh_shapes = sub_logical_mesh_shapes
    global_config.submesh_autosharding_global_configs = submesh_autosharding_global_configs
    global_config.logical_mesh_search_space = logical_mesh_search_space
    global_config.auto_stage_construction_imbalance_tolerance = auto_stage_construction_imbalance_tolerance
    global_config.pipeline_parallel_schedule = pipeline_parallel_schedule
    global_config.use_hlo_cost_model = use_hlo_cost_model
    global_config.profiling_database_filename = profiling_database_filename


is_worker = os.environ.get("PARAX_IS_WORKER", "False") == "True"

# Don't let the compilation on the driver node use GPUs.
# TODO(lmzheng): enable auto-tuning for compilation on workers.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS",
                                         "") + " --xla_gpu_autotune_level=0"

#os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=true"
