"""All global configurations for this project."""
import os


class GlobalConfig:
    """Global configuration of parax."""

    def __init__(self):
        ########## Options for @parallelize decorator ########## 
        self.devices = None
        self.strategy = "auto_sharding_parallel"
        self.memory_budget_per_device = None
        self.enable_profiling_communiation = False
        self.enable_mesh_shape_search = False
        self.cache_folder = "~/.parax"

        ########## Options for benchmark ########## 
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False


global_config = GlobalConfig()


def set_parallelize_options(devices=None,
                            strategy="auto_sharding_parallel",
                            memory_budget_per_device=None,
                            enable_profiling_communiation=False,
                            enable_mesh_shape_search=False,
                            cache_folder="~/.parax/"):
    """Set the global options for all @parallelize decorator.
    
    Args:
      devices: The device cluster.
      strategy: The parallelization strategy.
        Possible choices: {"auto_sharding_parallel",
        "pmap_data_parallel", "shard_data_parallel",
        "pipeline_parallel", "distributed_pipeline_parallel", "3d_parallel"}.
      memory_budget_per_device: The memory budget of one device in bytes.
      enable_profiling_communiation: Whether to enable the profiling communication
        stage before the search.
      enable_mesh_shape_search: Whether to include the choices of mesh_shape into
        the search space.
      cache_folder: The folder to store cached profiling results and strategies.
    """
    global global_config

    global_config.devices = devices
    global_config.strategy = strategy
    global_config.memory_budget_per_device = memory_budget_per_device
    global_config.enable_profiling_communiation = enable_profiling_communiation
    global_config.enable_mesh_shape_search = enable_mesh_shape_search
    global_config.cache_folder = cache_folder


# Don't let the compilation on the driver node use GPUs.
# TODO(lmzheng): enable auto-tuning for compilation on workers.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
