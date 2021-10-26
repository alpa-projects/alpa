"""All global configurations for this project."""
import os
import copy


class GlobalConfig:
    """Global configuration of parax.

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

        # logical mesh shape related options
        self.search_logical_mesh_shape = False
        self.mesh_shape_search_mode = "cost_model"
        self.mesh_shape_search_log_file = None
        self.profile_communication = False

        self.cache_folder = "parax_cache"
        self.cache_auto_sharding_ilp_solution = False

        ########## Options for pipeline stage ##########
        self.pipeline_stage_mode = "uniform_layer_gpipe"

        ########## Options for auto-sharding solver ##########
        self.allow_all_gather = True  # Wether allow all-gather during re-sharding.
        self.allow_all_to_all = True  # Wether allow all-to-all during re-sharding.
        self.force_data_parallel = False  # Whether force to generate data-parallel
        self.prefer_reduce_scatter = False  # Prefer reduce-scatter over allreduce.
        self.allow_recompute_heavy_op = False  # Allow replicated dot computation.

        ########## Options for benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False

        ########## Options for logging ##########
        self.print_xla_compilation_time = False

    def backup(self):
        """Backup the configs."""
        return copy.copy(self.__dict__)

    def restore(self, saved_dict):
        """Restore the configs from a backup."""
        global_config.__dict__ = saved_dict


global_config = GlobalConfig()


def set_parallelize_options(devices=None,
                            strategy="shard_parallel",
                            memory_budget_per_device=None,
                            search_logical_mesh_shape=False,
                            mesh_shape_search_mode="cost_model",
                            mesh_shape_search_log_file=None,
                            num_micro_batches=None,
                            profile_communication=False,
                            cache_folder="parax_cache",
                            cache_auto_sharding_ilp_solution=False,
                            pipeline_stage_mode="uniform_layer_gpipe"):
    """
    Set the global options for all @parallelize decorator.

    Args:
      devices: The device cluster.
      strategy (str): The parallelization strategy.
        Possible choices: {"shard_parallel",
        "pmap_data_parallel", "shard_data_parallel",
        "local_pipeline_parallel", "3d_parallel"}.
      memory_budget_per_device (Optional[float]): The memory budget of one device in bytes.
      search_logical_mesh_shape (bool): Whether to include the choices of logical mesh shape
        into the search space.
      mesh_shape_search_mode (str): Whether to use cost model or real measurement to pick
        the logical mesh shape. Possible choices: {"cost_model", "measurement"}.
      mesh_shape_search_log_file (Optional[str]): The file to store measurement records of
        logical mesh shape search.
      num_micro_batches (int): The number of micro batches in pipeline parallel and gradient
        accumulation.
      profile_communication (bool): Whether to use the profiled communication cost
        for the auto-sharding pass.
      cache_folder (str): The folder to store cached profiling results and strategies.
      cache_auto_sharding_ilp_solution (bool): Whether to cache the ilp solution
        generated during auto-sharding pass.
      pipeline_stage_mode (str): The algorithm used to construct pipeline
        stages. Possible choice: {"uniform_layer_gpipe", "auto_gpipe"}
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


# Don't let the compilation on the driver node use GPUs.
# TODO(lmzheng): enable auto-tuning for compilation on workers.

is_worker = os.environ.get("PARAX_IS_WORKER", "False") == "True"

os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS",
                                         "") + " --xla_gpu_autotune_level=0"
#os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=true"
