"""Use the auto sharding pass in XLA.

The compilation passes and status of an HloModule:

UNOPTIMIZED
  |
  |  pre_spmd_partitioner passes
  |
  |  auto_sharding pass
  V
SHARDING_ANNOTATED
  |
  |  spmd partitioner pass
  V
SPMD_PARTITIONED
  |
  |  post_spmd_partitioner passes
  V
FULLY_OPTIMIZED
"""
import copy
import enum
import logging
import multiprocessing
import time
import traceback
from typing import Sequence, Optional, Union, Tuple
import warnings

import numpy as np
from jax._src.lib import xla_bridge as xb, xla_client as xc, xla_extension as xe
from jax.core import ShapedArray
from jax.interpreters import xla, pxla

from alpa.global_env import global_config, AutoShardingOption
from alpa.measure_record import (MeasureInput, MeasureResult, StrategyConfig,
                                 save_to_file, SearchTask)
from alpa.util import check_arithmetic_sequence, get_compile_options, to_int_tuple, XlaPassContext

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# A constant to represent infinity
INFINITY_COST = 1e13


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

    def make_tile_spec(self, array, tensor_dims, mesh_dims):
        shape = array.shape
        sharding = [
            pxla.NoSharding(),
        ] * len(shape)
        mesh_mapping = [
            None,
        ] * len(self.id_mesh.shape)

        for i, (tensor_dim, mesh_dim) in enumerate(zip(tensor_dims, mesh_dims)):
            sharding[tensor_dim] = pxla.Chunked([self.id_mesh.shape[mesh_dim]],)
            mesh_mapping[mesh_dim] = pxla.ShardedAxis(i)

        for i, _ in enumerate(mesh_mapping):
            if mesh_mapping[i] is None:
                mesh_mapping[i] = pxla.Replicated(self.id_mesh.shape[i])

        return pxla.ShardingSpec(sharding, mesh_mapping)

    def __hash__(self):
        return hash((self.flatten_ids, self.id_mesh.shape, self.mesh_alpha,
                     self.mesh_beta))

    def __eq__(self, other):
        return ((self.flatten_ids, self.id_mesh.shape, self.mesh_alpha,
                 self.mesh_beta) == (other.flatten_ids, other.id_mesh.shape,
                                     other.mesh_alpha, other.mesh_beta))


def run_auto_sharding_pass(
        xla_computation: xe.XlaComputation,
        avals: Sequence[ShapedArray],
        out_avals: Sequence[ShapedArray],
        donated_invars: Sequence[bool],
        logical_mesh: LogicalDeviceMesh,
        return_mode: str,
        num_micro_batches: int,
        as_option: AutoShardingOption,
        rewrite_for_grad_acc: bool = False,
        rewrite_grad_acc_indices: Optional[Sequence[int]] = None,
        memory_budget_per_device: Optional[float] = None):
    """Run the auto-sharding pass to annotate sharding specs for an XLA Computation.

    Args:
      xla_computation: The xla computation got by tracing the jax function,
        whose status should be UNOPTIMIZED.
      avals: The abstract values of input arguments.
      out_avals: The abstract values of outputs.
      donated_invars: Whether the arguments are donated.
      logical_mesh: The logical device mesh.
      return_mode: The mode of return value.
        The choices are {"single", "stage_protos", "stage_and_hook_protos"}.
        If it is "single", return a single HLO Module proto, whose status is SPMD_PARTITIONED.
        If it is "stage_protos", return HLO Module protos of multiple pipeline stages,
          whose statuses are SHARDING_ANNOTATED.
        If it is "stage_and_hook_protos", return HLO Module protos of multiple pipeline stages
          and the hooked hlo sharding. The statuses of the returned protos are SHARDING_ANNOTATED.
      num_micro_batches: The number of micro batches
        if gradient accumulation is used. If this is set, the cost of all-reduce
        for gradient synchronization is divided by this number.
      as_option: The options of the auto-sharding solver.
      rewrite_for_grad_acc: Whether to do rewriting for gradient accumulation.
      rewrite_grad_acc_indices: The indices of tensors in output that are gradients.
      memory_budget_per_device: The memory budget per device in bytes.
    """
    global last_s_val
    global last_objective

    # Set compile options
    if memory_budget_per_device is None:
        memory_budget_per_device = -1

    if return_mode in ["stage_protos", "stage_and_hook_protos"]:
        multiple_stages = True
    else:
        multiple_stages = False

    backend = xb.get_backend("gpu")
    num_devices = logical_mesh.num_devices
    build_random_seed = global_config.build_random_seed
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed)

    # Set configs for force_zero_stage_3
    if as_option.force_zero_stage_3:
        # Generate a strategy similar to ZeRO stage 3
        force_data_parallel = True
        prefer_reduce_scatter = True
        reduce_scatter_aggressive_partition = True
        all_gather_threshold = as_option.force_zero_stage_3_all_gather_threshold
    else:
        # Use default settings
        force_data_parallel = as_option.force_data_parallel
        prefer_reduce_scatter = as_option.prefer_reduce_scatter
        reduce_scatter_aggressive_partition = False
        all_gather_threshold = 1 << 60

    # Set configs for force_data_parallel
    mesh_shape = logical_mesh.shape
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

    # Set configs for gradient accumulation rewrite pass
    if rewrite_for_grad_acc and rewrite_grad_acc_indices is None:
        rewrite_grad_acc_indices = tuple(
            range(
                len(xla_computation.program_shape().result_shape().tuple_shapes(
                ))))

    # Temporarily disable this.
    grad_acc_num_micro_batches = None

    with XlaPassContext({
            # Auto-sharding solver options
            "auto_sharding::memory_budget_per_device": memory_budget_per_device,
            "auto_sharding::force_all_gather_cost": not allow_all_gather,
            "auto_sharding::all_gather_cost": INFINITY_COST,
            "auto_sharding::force_all_to_all_cost": not allow_all_to_all,
            "auto_sharding::all_to_all_cost": INFINITY_COST,
            "auto_sharding::allow_replicated_parameters":
                as_option.allow_replicated_parameters,
            "auto_sharding::prefer_reduce_scatter": prefer_reduce_scatter,
            "auto_sharding::reduce_scatter_grad_acc_friendly": reduce_scatter_grad_acc_friendly,
            "auto_sharding::reduce_scatter_aggressive_partition": reduce_scatter_aggressive_partition,
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

            # Gradient accumulation rewrite
            "auto_sharding::rewrite_for_grad_acc": rewrite_for_grad_acc,
            "auto_sharding::rewrite_indices": rewrite_grad_acc_indices,

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
        compiled_module = xe.run_auto_sharding(xla_computation, compile_options)

    if multiple_stages:
        hlo_stage_names, hlo_stages = get_auto_sharded_hlo_stages()
        hooked_proto = get_hooked_sharding_protos()

    strategy_config = StrategyConfig(build_random_seed, logical_mesh.shape,
                                     all_gather_threshold,
                                     as_option.all_reduce_threshold, last_s_val,
                                     last_objective)

    if return_mode == "single":
        return compiled_module, strategy_config
    elif return_mode == "stage_protos":
        return hlo_stage_names, hlo_stages, strategy_config
    elif return_mode == "stage_and_hook_protos":
        return hlo_stage_names, hlo_stages, hooked_proto, strategy_config
    else:
        raise ValueError("Invalid return mode:" + return_mode)


def run_spmd_partitioner_pass(
        xla_computation: xe.XlaComputation,
        num_devices: int,
        rewrite_for_grad_acc: bool = False,
        rewrite_grad_acc_indices: Optional[Sequence[int]] = None):
    """Run SPMD partitioner pass on a sharding annotated HLO Module.

    Args:
      xla_computation: The input HLO Module. The status should be SHARDING_ANNOTATED.
      num_devices: The total number of devices.
      rewrite_for_grad_acc: Whether to do rewriting for gradient accumulation.
      rewrite_grad_acc_indices: The indices of tensors in output that are gradients.
    """
    backend = xb.get_backend("gpu")
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=global_config.build_random_seed)

    if rewrite_for_grad_acc and rewrite_grad_acc_indices is None:
        rewrite_grad_acc_indices = tuple(
            range(
                len(xla_computation.program_shape().result_shape().tuple_shapes(
                ))))

    with XlaPassContext({
            # Gradient accumulation rewrite
            "auto_sharding::rewrite_for_grad_acc": rewrite_for_grad_acc,
            "auto_sharding::rewrite_indices": rewrite_grad_acc_indices,
    }):
        compiled_module = xe.run_spmd_partitioner(xla_computation, compile_options)

    return compiled_module


def run_backend_compilation(backend: xe.Client,
                            xla_computation: Union[xe.XlaComputation,
                                                   xe.HloModule, bytes],
                            strategy_config: StrategyConfig,
                            num_devices: int):
    """Compile a spmd partitioned Hlo Module to an XLA executable.

    Args:
      backend: The XLA backend client.
      xla_computation: The input HLO Module. The status should be SPMD_PARTITIONED.
      strategy_config: The auto-sharding strategy solution.
      num_devices: The total number of devices.
      bypass_device_assignment_check: Whether to compile without exact devices.
    """
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=False,
        parameter_is_tupled_arguments=False,
        build_random_seed=strategy_config.build_random_seed)

    if isinstance(xla_computation, xe.HloModule):
        xla_computation = xe.XlaComputation(
            xla_computation.as_serialized_hlo_module_proto())
    elif isinstance(xla_computation, bytes):  # protobuf
        xla_computation = xe.XlaComputation(xla_computation)
    else:
        assert isinstance(xla_computation, xe.XlaComputation)

    with XlaPassContext({
            # Communication combiner options
            "combiner::all_gather_threshold":
                strategy_config.all_gather_threshold,
            "combiner::all_reduce_threshold":
                strategy_config.all_reduce_threshold,
            "combiner::use_continuous_buffer": True,
    }):
        compiled = backend.compile(xla_computation, compile_options)

    return compiled


def get_input_output_sharding_specs(
    hlo_module: xe.HloModule, avals: Sequence[ShapedArray],
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
    if sharding_type == xc.OpSharding.Type.OTHER:
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
    elif sharding_type == xc.OpSharding.Type.REPLICATED:
        sharding = (pxla.NoSharding(),) * len(aval.shape)
        mesh_mapping = (pxla.Replicated(np.prod(logical_mesh.shape)),)
    else:
        raise NotImplementedError("Type: " + str(sharding_type))

    return pxla.ShardingSpec(sharding, mesh_mapping)


def hlo_sharding_to_sharding_spec(
        hlo_sharding: xe.HloSharding, aval: ShapedArray,
        logical_mesh_shape: Sequence[int]) -> pxla.ShardingSpec:
    """Convert hlo sharding to sharding spec."""
    logical_mesh = LogicalDeviceMesh(
        None,
        np.arange(np.prod(logical_mesh_shape)).reshape(logical_mesh_shape))
    proto = hlo_sharding.proto_tuple()
    sharding_type, tuple_shardings = proto.type, proto.tuple_shardings
    if sharding_type == xc.OpSharding.Type.TUPLE:
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
    time_limit = 600
    assert "GLPK_CMD" in pulp.listSolvers(onlyAvailable=True), (
        "Please install ILP solvers by 'sudo apt install coinor-cbc glpk-utils'"
    )

    with warnings.catch_warnings():  # disable CBC warnings
        warnings.simplefilter("ignore")
        solver = pulp.COIN_CMD(mip=True,
                               msg=msg,
                               timeLimit=time_limit,
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

    last_s_val = s_val
    last_objective = objective

    if objective > INFINITY_COST:
        warnings.warn("Detect unexpected behaviors in the auto-sharding pass.")

    return s_val, e_val, objective, status


# Auto-sharded pipeline stages.
# These global variables are used to receive values from XLA c++ passes.
auto_sharded_hlo_stage_names = None
auto_sharded_hlo_stages = None

hooked_sharding_protos = None


def set_auto_sharded_hlo_stages(stages: Tuple[Sequence[str], Sequence[bytes]]):
    """Set the sliced auto-sharded stages. This is called in XLA SliceAutoShardedStages pass."""
    hlo_module_names, hlo_module_protos = stages
    global auto_sharded_hlo_stage_names, auto_sharded_hlo_stages
    auto_sharded_hlo_stage_names = hlo_module_names
    auto_sharded_hlo_stages = hlo_module_protos


def set_hooked_sharding_protos(hlo_module_proto: bytes):
    global hooked_sharding_protos
    hooked_sharding_protos = hlo_module_proto


def get_auto_sharded_hlo_stages() -> Sequence[bytes]:
    """Get the sliced hlo stages from the SliceAutoShardedStages pass."""
    return auto_sharded_hlo_stage_names, auto_sharded_hlo_stages


def get_hooked_sharding_protos() -> bytes:
    return hooked_sharding_protos
