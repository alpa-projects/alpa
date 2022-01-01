"""Use the auto sharding pass in XLA."""
import copy
import enum
import logging
import multiprocessing
import time
import traceback
from typing import Sequence, Optional, Union, Tuple
from warnings import warn

import numpy as np
from jax.core import ShapedArray
from jax.interpreters import xla, pxla
from jaxlib import xla_extension
from jaxlib.xla_client import OpSharding

from parax.global_env import global_config
from parax.measure_record import (MeasureInput, MeasureResult, StrategyConfig,
                                  save_to_file, SearchTask)
from parax.util import check_arithmetic_sequence, get_compile_options, to_int_tuple, XlaPassContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# A constant to represent infinity
INFINITY_COST = 1e13


class HloProtoStatus(enum.IntEnum):
    """The status of a HLO protobuf."""

    UNOPTIMIZED = 0  # An unoptimized HLO got from tracing the jaxpr.
    SHARDING_ANNOTATED = 1  # A HLO with sharding annotation attached.
    FULLY_OPTIMIZED = 2  # A fully optimized HLO which is already partitioned by
    # the SPMD partitioner.


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
        ret = copy.copy(self)
        for k, v in new_values.items():
            assert hasattr(ret, k)
            setattr(ret, k, v)
        return ret


class LogicalDeviceMesh:
    """
    A logical view of a physical mesh. The logical view is used in the auto-sharding pass.

    A physical mesh can have multiple logical views. (e.g., a 2x8 physical mesh can be viewed
    as a 1x16 or a 4x4 logical mesh). Each mesh dimension has its own latency and bandwidth.
    We use alpha-beta model to model the communication cost.
    """

    def __init__(self, physical_mesh, id_mesh, mesh_alpha=None, mesh_beta=None):
        self.physical_mesh = physical_mesh
        self.id_mesh = np.array(id_mesh)
        self.flatten_ids = tuple(int(x) for x in self.id_mesh.flatten())
        self.is_multi_host = False

        # coefficient for alpha-beta communication model
        if mesh_alpha is None:
            mesh_alpha = [1] * len(self.id_mesh.shape)
        if mesh_beta is None:
            mesh_beta = [1] * len(self.id_mesh.shape)
        self.mesh_alpha = tuple(mesh_alpha)
        self.mesh_beta = tuple(mesh_beta)

    @property
    def shape(self):
        return self.id_mesh.shape

    @property
    def num_devices(self):
        return np.prod(self.id_mesh.shape)

    def all_gather_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] *
                (num_devices - 1) / num_devices * num_bytes + 0.1)

    def all_reduce_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] * 2 *
                (num_devices - 1) / num_devices * num_bytes + 0.01)

    def reduce_scatter_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] *
                (num_devices - 1) / num_devices * num_bytes + 0.001)

    def all_to_all_cost(self, num_bytes, mesh_dim):
        num_devices = self.id_mesh.shape[mesh_dim]
        penalty_factor = num_devices / 2.0
        return (self.mesh_alpha[mesh_dim] + self.mesh_beta[mesh_dim] *
                (num_devices - 1) / num_devices / num_devices * num_bytes *
                penalty_factor + 0.001)

    def __hash__(self):
        return hash((self.flatten_ids, self.id_mesh.shape, self.mesh_alpha,
                     self.mesh_beta))

    def __eq__(self, other):
        return ((self.flatten_ids, self.id_mesh.shape, self.mesh_alpha,
                 self.mesh_beta) == (other.flatten_ids, other.id_mesh.shape,
                                     other.mesh_alpha, other.mesh_beta))


def compile_with_search(
        backend: xla_extension.Client,
        xla_computation: xla_extension.XlaComputation,
        avals: Sequence[ShapedArray], out_avals: Sequence[ShapedArray],
        donated_invars: Sequence[bool],
        logical_mesh_choices: Sequence[LogicalDeviceMesh],
        return_mode: str,
        num_micro_batches: int,
        as_option: AutoShardingOption,
        bypass_device_assignment_check: bool,
        memory_budget_per_device: Optional[float] = None,
        logical_mesh_search_mode: Optional[str] = None,
        logical_mesh_search_physical_mesh: Optional["PhysicalDeviceMesh"] = None,
        search_task: Optional[SearchTask] = None,
        record_file: Optional[str] = None):
    """Compile an XLA computation with mesh shape search and auto sharding solver.

    Args:
      backend: The XLA backend client.
      xla_computation: The unoptimized xla computation
        got by tracing the jax function.
      avals: The abstract values of input arguments.
      out_avals: The abstract values of outputs.
      donated_invars: Whether the arguments are donated.
      logical_mesh_choices: The candidates of logical mesh shape.
        If there is only one choice, use the given one. If there are multiple choices,
        we will try all of them and pick the best.
      return_mode: The mode of return value. The choices are {"executable", "stage_protos", "stage_and_hook_protos"}.
        If it is "executable", return the compiled xla executable.
        If it is "stage_protos", return the HLO Module proto of multiple pipeline stages.
        If it is "stage_and_hook_protos", return the HLO Module proto of multiple pipeline stages and the hooked hlo sharding.
      num_micro_batches: The number of micro batches
        if gradient accumulation is used. If this is set, the cost of all-reduce
        for gradient synchronization is divided by this number.
      as_option: The options of the auto-sharding solver.
      bypass_device_assignment_check: Whether to compile without exact devices.
      memory_budget_per_device: The memory budget per device in bytes.
      logical_mesh_search_mode: Only used when doing logical mesh shape search.
        The choices are {"measurement", "cost_model"}.
        If it is "measurement", use real profiling to pick the best logical mesh shape.
        If it is "cost_model", use cost estimation in HLO IR to pick the best one.
        This is ignored if len(logical_mesh_choices) == 1.
      logical_mesh_search_physical_mesh: Only used when doing logical mesh shape search.
        The physical device mesh used for logical mesh search.
      search_task: Only used when doing logical mesh shape search.
        Used when dumping measurement records to the file.
      record_file: Only used when doing logical mesh shape search.
        If is not None, dump measurement records into this file.
    """
    from parax import testing

    # Set compile options
    if memory_budget_per_device is None:
        memory_budget_per_device = -1

    if return_mode in ["stage_protos", "stage_and_hook_protos"]:
        multiple_stages = True
    else:
        multiple_stages = False

    run_backend_codegen = not bypass_device_assignment_check and not multiple_stages
    return_after_slice_auto_sharded_stages = bool(multiple_stages)

    num_devices = logical_mesh_choices[0].num_devices
    build_random_seed = global_config.build_random_seed
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed)

    def _invoke_compilation(logical_mesh):
        global last_s_val
        global last_objective

        mesh_shape = logical_mesh.shape

        # Set configs for force_zero_stage_3
        if as_option.force_zero_stage_3:
            # Generate a strategy similar to ZeRO stage 3
            force_data_parallel = True
            prefer_reduce_scatter = True
            reduce_scatter_aggresive_partition = True
            all_gather_threshold = as_option.force_zero_stage_3_all_gather_threshold
        else:
            # Use default settings
            force_data_parallel = as_option.force_data_parallel
            prefer_reduce_scatter = as_option.prefer_reduce_scatter
            reduce_scatter_aggresive_partition = False
            all_gather_threshold = 1 << 60

        # Set configs for force_data_parallel
        if force_data_parallel:
            # Forcibly generate data-parallel strategy
            allow_all_gather = False
            allow_all_to_all = False

            if mesh_shape[0] == 1:
                force_batch_dim_to_mesh_dim = 1
            elif mesh_shape[1] == 1:
                force_batch_dim_to_mesh_dim = 0
            else:
                raise ValueError(
                    f"Cannot force data parallel for the mesh shape {mesh_shape}. "
                    "Please make sure the mesh shape only has a single non-one dimension."
                )
        else:
            # Use default settings
            allow_all_gather = as_option.allow_all_gather
            allow_all_to_all = as_option.allow_all_to_all

            if as_option.force_batch_dim_to_mesh_dim is None:
                # Automatically set force_batch_dim_to_mesh_dim
                if logical_mesh.shape[0] > 1 and logical_mesh.shape[1] > 1:
                    # In 2d mesh, force the batch tensor dim to match the first mesh dim
                    force_batch_dim_to_mesh_dim = 0
                else:
                    force_batch_dim_to_mesh_dim = -1
            else:
                force_batch_dim_to_mesh_dim = as_option.force_batch_dim_to_mesh_dim

        # Set configs for reduce-scatter
        if num_micro_batches is not None and num_micro_batches > 1:
            reduce_scatter_grad_acc_friendly = True
        else:
            reduce_scatter_grad_acc_friendly = False

        # Temporarily disable this.
        grad_acc_num_micro_batches = None

        with XlaPassContext({
                # Build options
                "build_option::bypass_device_assignment_check": bypass_device_assignment_check,
                "build_option::run_backend_codegen": run_backend_codegen,
                "build_option::return_after_slice_auto_sharded_stages": return_after_slice_auto_sharded_stages,

                # Auto-sharding solver options
                "auto_sharding::enable": True,
                "auto_sharding::memory_budget_per_device": memory_budget_per_device,
                "auto_sharding::force_all_gather_cost": not allow_all_gather,
                "auto_sharding::all_gather_cost": INFINITY_COST,
                "auto_sharding::force_all_to_all_cost": not allow_all_to_all,
                "auto_sharding::all_to_all_cost": INFINITY_COST,
                "auto_sharding::allow_replicated_parameters":
                    as_option.allow_replicated_parameters,
                "auto_sharding::prefer_reduce_scatter": prefer_reduce_scatter,
                "auto_sharding::reduce_scatter_grad_acc_friendly": reduce_scatter_grad_acc_friendly,
                "auto_sharding::reduce_scatter_aggresive_partition": reduce_scatter_aggresive_partition,
                "auto_sharding::batch_matmul_always_split_batch": True,
                "auto_sharding::allow_recompute_heavy_op":
                    as_option.allow_recompute_heavy_op,
                "auto_sharding::allow_mixed_mesh_shape":
                    as_option.allow_mixed_mesh_shape,
                "auto_sharding::grad_acc_num_micro_batches":
                    grad_acc_num_micro_batches or 1,
                "auto_sharding::force_batch_dim_to_mesh_dim": force_batch_dim_to_mesh_dim,
                "auto_sharding::force_simple_heuristic":
                    as_option.force_simple_heuristic,

                # Device mesh
                "auto_sharding::device_mesh_ids": logical_mesh.flatten_ids,
                "auto_sharding::device_mesh_shape": tuple(logical_mesh.shape),
                "auto_sharding::device_mesh_alpha": tuple(
                    float(x) for x in logical_mesh.mesh_alpha),
                "auto_sharding::device_mesh_beta": tuple(
                    float(x) for x in logical_mesh.mesh_beta),
                "auto_sharding::device_mesh_prof_result": getattr(
                    logical_mesh.physical_mesh, "prof_result", None),

                # Communication combiner options
                "combiner::all_gather_threshold": all_gather_threshold,
                "combiner::all_reduce_threshold": as_option.all_reduce_threshold,
                "combiner::use_continuous_buffer": True,

                # Debug options
                "auto_sharding::simplify_graph": True,
                "auto_sharding::print_strategy": False,
                "auto_sharding::force_strategy": False,
                "auto_sharding::force_strategy_inst_indices": [],
                "auto_sharding::force_strategy_stra_names": [],
        }):
            compiled = xla.backend_compile(backend, xla_computation,
                                           compile_options)
        return compiled, last_s_val, last_objective

    if len(logical_mesh_choices) == 1:  # Compile with the given logical mesh
        logical_mesh = logical_mesh_choices[0]
        compiled, solution_vector, objective = _invoke_compilation(logical_mesh)
        if multiple_stages:
            hlo_stages = get_auto_sharded_hlo_stages()
            hooked_proto = get_hooked_sharding_protos()
    else:  # Search for the best logical mesh
        from parax.mesh_executable import NormalMeshDriverExecutable
        assert not multiple_stages
        best_logical_mesh = best_compiled = best_solution_vector = best_objective = None
        best_time_cost = float("inf")
        for logical_mesh in logical_mesh_choices:
            compiled, solution_vector, objective = _invoke_compilation(
                logical_mesh)
            strategy_config = StrategyConfig(build_random_seed,
                                             logical_mesh.shape,
                                             as_option.all_reduce_threshold,
                                             solution_vector)

            if logical_mesh_search_mode == "measurement":
                mesh_exe = NormalMeshDriverExecutable(logical_mesh_search_physical_mesh,
                                                      compiled, strategy_config, avals,
                                                      out_avals, donated_invars)
                time_costs = tuple(mesh_exe.profile_with_dummy_inputs())
            else:
                assert logical_mesh_search_mode == "cost_model"
                time_costs = (objective,)

            if np.mean(time_costs) < best_time_cost:
                (best_logical_mesh, best_compiled, best_solution_vector,
                 best_objective) = (logical_mesh, compiled, solution_vector,
                                    objective)
                best_time_cost = np.mean(time_costs)

            # Save records to file
            if record_file is not None:
                assert search_task is not None
                inp = MeasureInput(search_task, strategy_config)
                res = MeasureResult(time_costs, objective, 0, int(time.time()))
                save_to_file([inp], [res], record_file)

        logical_mesh, compiled, solution_vector, objective = (
            best_logical_mesh, best_compiled, best_solution_vector,
            best_objective)

    testing.last_compiled_executable = compiled
    testing.last_compiled_auto_sharding_objective = objective
    strategy_config = StrategyConfig(build_random_seed, logical_mesh.shape,
                                     as_option.all_reduce_threshold,
                                     solution_vector)

    if return_mode == "executable":
        return compiled, strategy_config
    elif return_mode == "stage_protos":
        return hlo_stages, strategy_config
    elif return_mode == "stage_and_hook_protos":
        return hlo_stages, hooked_proto, strategy_config
    else:
        raise ValueError("Invalid return mode:" + return_mode)


def compile_with_given_strategy(
        backend: xla_extension.Client,
        xla_computation: xla_extension.XlaComputation,
        strategy_config: StrategyConfig,
        num_devices: int,
        hlo_proto_status: HloProtoStatus,
        bypass_device_assignment_check: bool,
        rewrite_for_grad_acc: bool = False,
        rewrite_grad_acc_indices: Optional[Sequence[int]] = None,
        run_backend_codegen: Union[str, bool] = "auto"):
    """Compile an XLA computation with a given auto sharding strategy.

    Args:
      backend: The XLA backend client.
      xla_computation: The unoptimized xla computation
        got by tracing the jax function.
      strategy_config: The auto-sharding strategy solution.
      num_devices: The total number of devices.
      bypass_device_assignment_check: Set this to true if this compilation is invoked
        on the driver node in the multi-host setting.
      hlo_proto_status: The optimization status of the
        input xla computation. see docs in the definition of `HloProtoStatus`.
      rewrite_for_grad_acc: Whether to do rewriting for gradient accumulation.
      rewrite_grad_acc_indices: The indices of tensors in output that are gradients.
      run_backend_codegen: Whether to run the backend codegen to generate cuda binaries.
    """
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=strategy_config.build_random_seed)
    logical_mesh_shape = strategy_config.logical_mesh_shape

    # Skip some compilation stages to
    # 1. accelerate the compilation.
    # 2. make sure the annotaed sharding is not modified by other passes.
    if hlo_proto_status == HloProtoStatus.UNOPTIMIZED:
        run_hlo_passes = True
        run_pre_spmd_partitioner_passes = True
        run_auto_sharding = True
        solution_vector = strategy_config.auto_sharding_solution_vector
    elif hlo_proto_status == HloProtoStatus.SHARDING_ANNOTATED:
        run_hlo_passes = True
        run_pre_spmd_partitioner_passes = False
        run_auto_sharding = False
        solution_vector = []
    elif hlo_proto_status == HloProtoStatus.FULLY_OPTIMIZED:
        run_hlo_passes = False
        run_auto_sharding = False
        run_pre_spmd_partitioner_passes = False
        solution_vector = []
    else:
        raise ValueError(f"Invalid status: {hlo_proto_status}")

    if run_backend_codegen == "auto":
        run_backend_codegen = not bypass_device_assignment_check
    else:
        assert isinstance(run_backend_codegen, bool)

    if rewrite_for_grad_acc and rewrite_grad_acc_indices is None:
        rewrite_grad_acc_indices = tuple(
            range(
                len(xla_computation.program_shape().result_shape().tuple_shapes(
                ))))

    with XlaPassContext({
            # Build options
            "build_option::bypass_device_assignment_check": bypass_device_assignment_check,
            "build_option::run_hlo_passes": run_hlo_passes,
            "build_option::run_pre_spmd_partitioner_passes": run_pre_spmd_partitioner_passes,
            "build_option::run_backend_codegen": run_backend_codegen,

            # Auto-sharding solver options
            "auto_sharding::enable": run_auto_sharding,
            "auto_sharding::load_solution_vector": True,
            "auto_sharding::solution_vector": to_int_tuple(solution_vector),
            "auto_sharding::rewrite_for_grad_acc": rewrite_for_grad_acc,
            "auto_sharding::rewrite_indices": rewrite_grad_acc_indices,

            # Device mesh
            "auto_sharding::device_mesh_ids": tuple(range(num_devices)),
            "auto_sharding::device_mesh_shape": tuple(logical_mesh_shape),

            # Communication combiner options
            "combiner::all_gather_threshold": 1 << 60,
            "combiner::all_reduce_threshold": strategy_config.all_reduce_threshold,
            "combiner::use_continuous_buffer": True,

            # Other useless but required arguments
            "auto_sharding::device_mesh_alpha":
                (1.0,) * len(logical_mesh_shape),
            "auto_sharding::device_mesh_beta": (1.0,) * len(logical_mesh_shape),
            "auto_sharding::device_mesh_prof_result": None,
    }):
        compiled = backend.compile(xla_computation, compile_options)

    return compiled


def get_input_output_sharding_specs(
    hlo_module: xla_extension.HloModule, avals: Sequence[ShapedArray],
    out_avals: Sequence[ShapedArray], num_devices: int,
    logical_mesh_shape: Sequence[int]
) -> Tuple[Sequence[pxla.ShardingSpec], Sequence[pxla.ShardingSpec]]:
    """Get the sharding specs of input/output tensors from an HloModule.

    Args:
      hlo_module: The sharded HLO module.
      avals: The abstract values of input tensors.
      out_avals: The abstract values of output tensors.
      num_devices: The total number of devices.
      logical_mesh_shape: The shape of logical mesh.

    Returns:
      input_sharding_specs: The sharding specs of input tensors.
      output_sharding_specs: The sharding specs of output tensors.
    """
    if num_devices != 1:
        input_shardings = hlo_module.spmd_parameters_shardings()
        input_sharding_specs = [
            hlo_sharding_to_sharding_spec(proto, aval, logical_mesh_shape)
            for (proto, aval) in zip(input_shardings, avals)
        ]
        output_shardings = hlo_module.spmd_output_sharding()
        output_sharding_specs = hlo_sharding_to_sharding_spec(
            output_shardings, out_avals, logical_mesh_shape)
    else:
        # The spmd partition related code will be bypassed if num_partitions == 1.
        # Assume all sharding specs are replicated.
        input_sharding_specs = [
            make_replicated_spec(aval, logical_mesh_shape) for aval in avals
        ]
        output_sharding_specs = [
            make_replicated_spec(aval, logical_mesh_shape) for aval in out_avals
        ]
    return input_sharding_specs, output_sharding_specs


def _hlo_sharding_to_sharding_spec_no_tuple(
        proto: bytes, aval: ShapedArray,
        logical_mesh: Sequence[int]) -> pxla.ShardingSpec:
    """The internal function of hlo_sharding_to_sharding_spec."""
    sharding_type, tile_assignment_dimensions, tile_assignment_devices = (
        proto.type, proto.tile_assignment_dimensions,
        proto.tile_assignment_devices)

    sharding = []
    mesh_mapping = []
    if sharding_type == OpSharding.Type.OTHER:
        tile_assignment = np.array(tile_assignment_devices).reshape(
            tile_assignment_dimensions)

        tile_dims = []
        for i in range(len(tile_assignment_dimensions)):
            if tile_assignment_dimensions[i] != 1:
                tile_dims.append(i)

        tile_dims_delta = []
        success = True
        for dim in tile_dims:
            indices = tuple(0 if i != dim else slice(None)
                            for i in range(tile_assignment.ndim))
            device_ids = tile_assignment[indices]
            delta = check_arithmetic_sequence(device_ids)
            if delta is None:
                success = False
                break
            tile_dims_delta.append(delta)

        if success:
            tile_dims_order = list(range(len(tile_dims)))
            tile_dims_order.sort(key=lambda i: -tile_dims_delta[i])

            ct = 0
            for i in range(len(aval.shape)):
                if tile_assignment_dimensions[i] == 1:
                    sharding.append(pxla.NoSharding())
                else:
                    sharding.append(
                        pxla.Chunked([tile_assignment_dimensions[i]]))
                    mesh_mapping.append(pxla.ShardedAxis(ct))
                    ct += 1

            if len(tile_dims) > len(mesh_mapping):
                # replicate on the last tile dim
                mesh_mapping.append(
                    pxla.Replicated(tile_assignment_dimensions[-1]))

            mesh_mapping = [mesh_mapping[idx] for idx in tile_dims_order]
        else:
            # The normal path fails, because one tensor dim is chunked into
            # mutliple parts. We only handle a special case here.
            assert len(aval.shape) == 1, "Only support 1d case"
            assert len(tile_assignment_dimensions) == len(aval.shape)
            for col in range(len(tile_assignment_devices)):
                if tile_assignment_devices[col] == 1:
                    break
            sharding = (pxla.Chunked(
                (tile_assignment_dimensions[0] // col, col)),)
            mesh_mapping = (pxla.ShardedAxis(1), pxla.ShardedAxis(0))
    elif sharding_type == OpSharding.Type.REPLICATED:
        sharding = (pxla.NoSharding(),) * len(aval.shape)
        mesh_mapping = (pxla.Replicated(np.prod(logical_mesh.shape)),)
    else:
        raise NotImplementedError("Type: " + str(sharding_type))

    return pxla.ShardingSpec(sharding, mesh_mapping)


def hlo_sharding_to_sharding_spec(
        hlo_sharding: xla_extension.HloSharding, aval: ShapedArray,
        logical_mesh_shape: Sequence[int]) -> pxla.ShardingSpec:
    """Convert hlo sharding to sharding spec."""
    logical_mesh = LogicalDeviceMesh(
        None,
        np.arange(np.prod(logical_mesh_shape)).reshape(logical_mesh_shape))
    proto = hlo_sharding.proto_tuple()
    sharding_type, tuple_shardings = proto.type, proto.tuple_shardings
    if sharding_type == OpSharding.Type.TUPLE:
        avals = aval
        return [
            _hlo_sharding_to_sharding_spec_no_tuple(shard, aval, logical_mesh)
            for (shard, aval) in zip(tuple_shardings, avals)
        ]
    else:
        return _hlo_sharding_to_sharding_spec_no_tuple(proto, aval,
                                                       logical_mesh)


def make_replicated_spec(
        aval: ShapedArray,
        logical_mesh_shape: Sequence[int]) -> pxla.ShardingSpec:
    """Make a replicated ShardingSpec."""
    sharding = (pxla.NoSharding(),) * len(aval.shape)
    mesh_mapping = (pxla.Replicated(np.prod(logical_mesh_shape)),)
    return pxla.ShardingSpec(sharding, mesh_mapping)


def call_solver_serialized_args(*args):
    """Call the solver with serialized arguments and handle python errors."""
    try:
        ret = _call_solver_serialized_args(*args)
    except AssertionError:
        ret = None
        info = str(traceback.format_exc()[:-1])
    except Exception:  # pylint: disable=broad-except
        ret = None
        info = str(traceback.format_exc()[:-1])

    if ret is None:
        print(info)

    return ret


# The last solution vector of auto sharding.
last_s_val = None

# The last objective value of the best ILP solution.
last_objective = None


# pylint: disable=import-outside-toplevel
def _call_solver_serialized_args(
        N,
        M,
        s_len_np,
        s_follow_np,
        E_np,
        A_np,
        L_np,  # noqa
        c_np,
        d_np,
        m_np,
        r_np,
        v_np,
        s_init_np=None):
    """Call the solver with serialized arguments."""
    global last_s_val, last_objective

    import pulp
    from pulp import LpVariable, LpProblem, LpMinimize, lpSum, lpDot, LpStatus
    tic = time.time()

    for x in [s_len_np, E_np, A_np, L_np, c_np, d_np, m_np, r_np, v_np]:
        assert isinstance(x, np.ndarray)
    assert len(s_len_np) == N, "s_len_np"

    # Dump arguments for re-solving
    # pickle.dump([N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
    #              c_np, d_np, m_np, r_np, v_np, s_init_np],
    #              open("args.pkl", "wb"))
    # TODO(lmzheng): cache the ILP solution.

    def get_non_zero_index(binary_vector):
        """Get the index of non-zero item in a vector."""
        ct = 0
        ret = None
        for i, elem in enumerate(binary_vector):
            if pulp.value(elem):
                ret = i
                ct += 1

        assert ct == 1
        return ret

    # 0. Unpack flatten numpy arrays
    s_len = s_len_np
    s_follow = s_follow_np

    E = E_np.reshape((-1, 2))
    r = []
    pt = 0
    edge_set = set()
    for (i, j) in E:
        prod_length = s_len[i] * s_len[j]

        if (i, j) in edge_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        edge_set.add((i, j))
        r.append(r_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(r_np)

    A = A_np.reshape((-1, 2))
    v = []
    pt = 0
    for (i, j) in A:
        prod_length = s_len[i] * s_len[j]
        v.append(v_np[pt:pt + prod_length])
        pt += prod_length
    assert pt == len(v_np)

    L = []
    pt = N
    for i in range(N):
        length = L_np[i]
        L.append(L_np[pt:pt + length])
        pt += length
    assert pt == len(L_np)

    c = []
    d = []
    m = []
    pt = 0
    for i in range(N):
        length = s_len[i]
        c.append(c_np[pt:pt + length])
        d.append(d_np[pt:pt + length])
        m.append(m_np[pt:pt + length])
        pt += length
    assert pt == len(c_np), f"{pt} == {len(c_np)}"
    assert pt == len(d_np), f"{pt} == {len(d_np)}"
    assert pt == len(m_np), f"{pt} == {len(m_np)}"

    # 1. Create variables
    s = []
    e = []

    num_nodes = 0
    reverse_follow_backpatch = []
    for i in range(N):
        if s_follow[i] < 0:
            if s_len[i] == 1:
                s.append([1])
            else:
                num_nodes += 1
                s.append(
                    LpVariable.matrix(f"s[{i}]", (range(s_len[i]),),
                                      cat="Binary"))
        else:
            if s_follow[i] < len(s):
                s.append(s[s_follow[i]])
            else:
                s.append(None)
                reverse_follow_backpatch.append(i)

    for i in reverse_follow_backpatch:
        s[i] = s[s_follow[i]]

    num_edges = 0
    for (idx, (i, j)) in enumerate(E):
        if len(s[i]) == 1:
            e.append(s[j])
        elif len(s[j]) == 1:
            e.append(s[i])
        else:
            num_edges += 1
            e.append(
                LpVariable.matrix(f"e[{i},{j}]",
                                  (range(len(s[i]) * len(s[j])),),
                                  cat="Binary"))
        assert len(e[idx]) == len(r[idx])

    # 2. Set initial value for warm start
    if s_init_np is not None:
        s_init = s_init_np.reshape((-1, 3))
        for (idx, value, fix) in s_init:
            for i in range(len(s[idx])):
                s[idx][i].setInitialValue(i == value)
                if fix:
                    s[idx][i].fixValue()

    # 3. Objective
    prob = LpProblem("myProblem", LpMinimize)
    # compute cost
    obj = 0
    for i in range(N):
        obj += lpDot(s[i], c[i]) + lpDot(s[i], d[i])

    # communication cost
    for i in range(len(E)):
        obj += lpDot(e[i], r[i])

    prob += obj

    # 4. Constraints
    # (a). specified by `cat="Binary"`

    # (b)
    for i in range(N):
        if s_follow[i] < 0:
            prob += lpSum(s[i]) == 1

    # (c)
    if M > 0:
        for t in range(N):
            mem = 0
            for i in L[t]:
                mem += lpSum(s[i][j] * m[i][j] for j in range(len(s[i])))
            prob += mem <= M

    # (d). specified by `cat="Binary"`

    for (idx, (i, j)) in enumerate(E):
        if s_len[i] == 1 or s_len[j] == 1:
            continue

        # (e)
        prob += lpSum(e[idx]) == 1

        # (f)
        for row in range(len(s[i])):
            C = len(s[j])
            prob += lpSum(
                e[idx][row * C + col] for col in range(0, C)) <= s[i][row]

        # (g)
        for col in range(len(s[j])):
            R = len(s[i])
            C = len(s[j])
            prob += lpSum(
                e[idx][row * C + col] for row in range(0, R)) <= s[j][col]

    # (h)
    alias_set = set()
    for (idx, (i, j)) in enumerate(A):
        R = len(s[i])
        C = len(s[j])
        if (i, j) in alias_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")

        alias_set.add((i, j))
        alias_set.add((j, i))

        for row in range(len(s[i])):
            for col in range(len(s[j])):
                if v[idx][row * C + col] > 0.5:
                    prob += s[i][row] + s[j][col] <= 1

    verbose = False

    msg = verbose
    time_limit = 2000
    assert "GLPK_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc glpk-utils'"
    )
    solver = pulp.COIN_CMD(
        mip=True,
        msg=msg,
        #timeLimit=time_limit,
        threads=multiprocessing.cpu_count())
    # solver = pulp.GLPK_CMD(mip=True, msg=msg, timeLimit=time_limit)
    prob.solve(solver)

    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    if verbose:
        print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
              f"Time: {time.time() - tic}")
        print(f"#nodes: {num_nodes},  #edges: {num_edges}")

    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError(
            "Cannot run the function under the given memory budget. "
            "Please increase the memory budget.")

    # Get and check results
    s_val = np.full((N,), -1, dtype=np.int32)
    for i in range(N):
        s_val[i] = get_non_zero_index(s[i])

    e_val = np.full((len(E),), -1, dtype=np.int32)
    for (idx, (i, j)) in enumerate(E):
        e_val[idx] = get_non_zero_index(e[idx])
        i_spec_index = e_val[idx] // len(s[j])
        j_spec_index = e_val[idx] % len(s[j])
        assert i_spec_index == s_val[i], f"e_val[{i}][{j}]"
        assert j_spec_index == s_val[j], f"e_val[{i}][{j}]"
        if verbose and r[idx][e_val[idx]] > 0:
            print(f"Edge cost {(i, j)} : {r[idx][e_val[idx]]}")

    last_objective = objective
    last_s_val = s_val

    if objective > INFINITY_COST:
        warn("Detect unexpected behaviors in the auto-sharding pass.")

    return s_val, e_val, objective, status


# Auto-sharded pipeline stages.
# These global variables are used to receive values from XLA c++ passes.
auto_sharded_hlo_stages = None

hooked_sharding_protos = None


def set_auto_sharded_hlo_stages(hlo_module_protos: Sequence[bytes]):
    """Set the sliced auto-sharded stages. This is called in XLA SliceAutoShardedStages pass."""
    global auto_sharded_hlo_stages
    auto_sharded_hlo_stages = hlo_module_protos


def set_hooked_sharding_protos(hlo_module_proto: bytes):
    global hooked_sharding_protos
    hooked_sharding_protos = hlo_module_proto


def get_auto_sharded_hlo_stages() -> Sequence[bytes]:
    """Get the sliced hlo stages from the SliceAutoShardedStages pass."""
    return auto_sharded_hlo_stages


def get_hooked_sharding_protos() -> bytes:
    return hooked_sharding_protos
