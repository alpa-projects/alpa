"""Alpa is a system for training large-scale neural networks."""

# Check version first
from alpa.version import __version__

# Import all public packages
from . import api
from . import collective
from . import create_state_parallel
from . import device_mesh
from . import data_loader
from . import global_env
from . import follow_parallel
from . import mesh_profiling
from . import monkey_patch
from . import parallel_method
from . import parallel_plan
from . import pipeline_parallel
from . import shard_parallel
from . import timer
from . import util

# Short cuts
from alpa.api import (init, shutdown, parallelize, grad, value_and_grad,
                      clear_executable_cache)
from alpa.data_loader import DataLoader, MeshDriverDataLoader
from alpa.device_mesh import (
    DeviceCluster, PhysicalDeviceMesh, LocalPhysicalDeviceMesh,
    DistributedPhysicalDeviceMesh, DistributedArray, prefetch,
    get_global_cluster, get_global_physical_mesh,
    set_global_virtual_physical_mesh, set_seed, get_global_num_devices)
from alpa.global_env import global_config
from alpa.mesh_profiling import ProfilingResultDatabase
from alpa.parallel_method import (ShardParallel, PipeshardParallel,
                                  DataParallel, Zero2Parallel, Zero3Parallel,
                                  CreateStateParallel, FollowParallel)
from alpa.parallel_plan import plan_to_method
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary
from alpa.pipeline_parallel.layer_construction import (manual_remat,
                                                       automatic_remat,
                                                       ManualLayerOption,
                                                       AutoLayerOption)
from alpa.pipeline_parallel.stage_construction import (ManualStageOption,
                                                       AutoStageOption,
                                                       UniformStageOption)
from alpa.shard_parallel.auto_sharding import AutoShardingOption
from alpa.serialization import save_checkpoint, restore_checkpoint
from alpa.timer import timers
