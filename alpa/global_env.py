"""All global configurations for this project."""
import os


class GlobalConfig:
    """The global configuration of alpa."""

    def __init__(self):
        ########## Options of device mesh ##########
        # See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
        self.xla_client_mem_fraction = float(
            os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", 0.9))
        self.xla_gpu_autotune_level = 4
        # The threshold to tigger a batched deletion on workers.
        self.delete_remote_arrays_threshold = 50
        # Whether to use AWS EFA network interface
        self.use_aws_efa = os.environ.get("ALPA_USE_AWS_EFA",
                                          "").lower() in ["true", "1"]
        # Random seed used for compilation
        self.compile_random_seed = 42
        # Random seed used for runtime
        self.runtime_random_seed = 42

        ########## Options of shard_parallel ##########
        # Whether to sync before and after the executable for accurate internal
        # timer
        self.shard_parallel_sync_for_timer = False

        ########## Options of pipeline_parallel ##########
        # Whether to debug with pipeshard runtime. If turned on, no physical
        # resource is required until launching PipeshardExecutable.
        self.debug_with_pipeshard_runtime = False
        # Whether to use the whole cluster for stage profiling. If not, only
        # use the given mesh.
        self.profile_with_whole_ray_cluster = True
        # Stage construction profiling time threshold.
        self.profile_timeout = 500
        # Stage construction profiling retry threshold.
        # Some communication patterns may meet deadlock, so it needs retry.
        self.profile_maximum_retry = 2
        # Whether to forcely set stage construction's submesh choices
        self.overwrite_submesh_choices = None
        self.always_donate_micro_batch_vars = True

        ########## Options of pipeline runtime ##########
        self.pipeline_check_alive = True
        # Whether to sync before and after the executable for accurate internal
        # timer
        self.pipeline_sync_for_timer = False
        # Whether to use distributed compilation in pipeline parallel for
        # each stage. Disabling it helps debug.
        self.pipeline_distributed_compile = True
        # Whether to use single-byte signal tensor for send/recv.
        # This is a debug option.
        self.pipeline_use_signal_send_recv = False
        # Whether to use the scatter-gater/local-all-gather optimization.
        self.use_local_allgather = True
        self.eagerly_create_communicators = True
        self.use_memzero_for_gradient_accumulation = False
        # Cross mesh resharding mode. Possible choices: {"send_recv",
        # "broadcast"}
        self.resharding_mode = "send_recv"
        # Which nccl to use. Possible choices: {"cupy",
        # "xla_extension"}
        self.nccl_mode = "cupy"

        ########## Options of XLA compilation ##########
        # Whether to use xla while instruction for preventing CSE in
        # rematerialization
        self.remat_using_while = False

        ########## Options of benchmark ##########
        # If true, the system is allowed to use dummy values during
        # tensor creation and copy to reduce the initialization and copy time.
        # This will produce wrong results but is acceptable for
        # data-independent benchmarks.
        self.use_dummy_value_for_benchmarking = False

        ########## Options of logging ##########
        self.print_compilation_time = False
        self.print_auto_layer_stats = False


global_config = GlobalConfig()

# Other environment setup
is_worker = os.environ.get("ALPA_IS_WORKER", "False") == "True"

os.environ["XLA_FLAGS"] = os.environ.get(
    "XLA_FLAGS", "") + " --xla_gpu_enable_async_all_reduce=false"
