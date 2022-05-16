"""All global configurations for this project."""
import os


class GlobalConfig:
    """The global configuration of alpa."""

    def __init__(self):
        ########## Options of device mesh ##########
        self.xla_client_mem_fraction = float(
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.9))
        self.xla_gpu_autotune_level = 4
        self.delete_remote_buffers_threshold = 200
        # use AWS EFA network interface
        self.use_aws_efa = os.environ.get("ALPA_USE_AWS_EFA", "").lower() in [
            "true", "1"
        ]

        ########## Options of shard_parallel ##########
        self.shard_parallel_sync_for_timer = False

        ########## Options of pipeline_parallel ##########
        self.profile_with_whole_ray_cluster = True
        self.debug_with_local_runtime = False
        self.profile_timeout = 500
        self.profile_maximum_retry = 2
        self.overwrite_submesh_choices = None

        ########## Options of pipeline runtime ##########
        self.pipeline_check_alive = True
        # Whether to sync before and after the executable for accurate internal timer
        self.pipeline_sync_for_timer = False
        # Whether to use distributed compilation in pipeline parallel for
        # each stage. Disabling it helps debug.
        self.pipeline_distributed_compile = True
        self.pipeline_use_signal_send_recv = False
        self.precompile_resharding_tasks = True
        self.use_scatter_gather = True
        self.eagerly_create_communicators = True
        self.use_memzero_for_gradient_accumulation = False
        # Cross mesh resharding mode. Possible choices: {"send_recv", "broadcast"}
        self.resharding_mode = "send_recv"

        ########## Options of XLA compilation ##########
        self.build_random_seed = 42
        # Whether to use xla while instruction for preventing CSE in rematerialization
        self.remat_using_while = False

        ########## Options of benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False

        ########## Options of logging ##########
        self.print_compilation_time = False

        ########## Options of ray namespace ##########
        self.default_ray_namespace_prefix = "alpa-train"
        self.unittest_ray_namespace_prefix = "alpa-unittest"


global_config = GlobalConfig()

# Other environment setup
is_worker = os.environ.get("ALPA_IS_WORKER", "False") == "True"

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=false"
