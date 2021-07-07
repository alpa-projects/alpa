"""All global configurations for this project."""
import os


class GlobalConfig:
    """Global configuration of parax."""

    def __init__(self):
        # choices: {'data_parallel', 'auto_sharding'}
        self.shard_parallel_strategy = 'auto_sharding'

        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False


global_config = GlobalConfig()

# Don't let the compilation on the driver node use GPUs.
os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_autotune_level=0"
# TODO(lmzheng): enable auto-tuning for compilation on workers.
