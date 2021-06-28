"""Use the auto sharding pass in XLA."""
from enum import Enum, auto
import logging
import multiprocessing
import time
import traceback

from warnings import warn
import numpy as np
from jax import linear_util as lu
from jax._src.util import (partial, extend_name_stack, wrap_name)
from jax.interpreters import xla, pxla, partial_eval as pe
from jax.lib import xla_bridge as xb, xla_client as xc
from jaxlib.xla_client import OpSharding

from parax import testing
from parax.device_mesh import LogicalDeviceMesh, PhysicalDeviceMesh
from parax.measure_record import StrategyConfig
from parax.xla_pass_context import XlaPassContext
from parax.util import to_int_tuple, get_compile_options

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# pylint: disable=too-many-arguments,too-many-locals
def auto_sharding_callable(   # noqa MC0001
        fun: lu.WrappedFun,
        in_tree,
        out_tree_thunk,
        donated_invars,
        physical_mesh,
        logical_mesh_search_mode,
	logical_mesh_choices,
        memory_budget_per_device,
        strategy_config,
        *avals):
    """Compile a callable with auto-sharding pass."""
    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)

    # Convert jaxpr to XLA HLO
    c = xb.make_computation_builder(f"auto_shard_{fun.__name__}")
    xla_consts = map(partial(xb.constant, c), consts)
    tuple_args = False
    xla_args, donated_invars = xla._xla_callable_args(c, avals, tuple_args, donated_invars=donated_invars)
    backend_name = 'gpu'
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
    transformed_name = fun.__name__
    out_nodes = xla.jaxpr_subcomp(
        c, jaxpr, backend_name, axis_env, xla_consts,
        extend_name_stack(wrap_name(transformed_name, 'auto_sharding')), *xla_args)
    out_tuple = xc.ops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    backend = xb.get_backend(backend_name)
    if backend.platform in ("gpu", "tpu"):
        donation_results = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
    if any(donation_results):
        # TODO(tomhennigan): At call time we should mark these buffers as deleted.
        unused_donations = [str(c.GetShape(a))
                            for a, d in zip(xla_args, donation_results) if d]
        warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))

    # Run XLA compliation
    built = c.Build(out_tuple)
    if strategy_config is None:
        compiled, strategy_config = compile_with_search(
            backend, built, physical_mesh,
            logical_mesh_search_mode, logical_mesh_choices, memory_budget_per_device,
        )
    else:
        compiled = compile_with_given_strategy(
            backend, built, physical_mesh.total_devices, physical_mesh.is_distributed,
            strategy_config
        )
    hlo_module = compiled.hlo_modules()[0]
    logical_mesh_shape = strategy_config.logical_mesh_shape

    # Send code and strategy to remote workers
    if physical_mesh.is_distributed:
        unoptimized_hlo_proto = built.as_serialized_hlo_module_proto()
        compiled = physical_mesh.compile_remote_executable(
            unoptimized_hlo_proto, strategy_config)

    # Read HloSharding from HloModule and convert them to ShardingSpec
    num_partitions = physical_mesh.total_devices
    if num_partitions != 1:
        input_shardings = hlo_module.spmd_parameters_shardings()
        input_sharding_specs = [
            hlo_sharding_to_sharding_spec(proto_tuple, aval, logical_mesh_shape)
            for (proto_tuple, aval) in zip(input_shardings, avals)
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

    # Return the final callable
    return physical_mesh.get_callable_with_arg_handler(compiled, avals, out_avals,
                                                       input_sharding_specs, output_sharding_specs,
                                                       donated_invars)

def compile_with_search(backend,
                        xla_computation,
                        physical_mesh,
                        logical_mesh_search_mode,
                        logical_mesh_choices,
                        memory_budget_per_device):
    """Compile an XLA computation with mesh shape search and auto sharding solver.""" 
    unoptimized_hlo_proto = xla_computation.as_serialized_hlo_module_proto()

    # Set compile options
    if memory_budget_per_device is None:
        memory_budget_per_device = -1
    if physical_mesh.is_distributed:
        by_pass_device_assignment_check = True
    else:
        by_pass_device_assignment_check = False

    build_random_seed = 42
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=physical_mesh.total_devices,
        device_assignment=np.arange(physical_mesh.total_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=build_random_seed
    )

    def _invoke_compilation(logical_mesh):
        global last_s_val
        global last_objective

        with XlaPassContext({
            # Solver options
            "auto_sharding::enable": True,
            "auto_sharding::memory_budget_per_device": memory_budget_per_device,
            "auto_sharding::force_all_gather_cost": False,
            "auto_sharding::all_gather_cost": 1e10,

            # Device mesh
            "auto_sharding::device_mesh_ids": logical_mesh.flatten_ids,
            "auto_sharding::device_mesh_shape": tuple(logical_mesh.id_mesh.shape),
            "auto_sharding::device_mesh_alpha": tuple(float(x) for x in logical_mesh.mesh_alpha),
            "auto_sharding::device_mesh_beta": tuple(float(x) for x in logical_mesh.mesh_beta),
            "auto_sharding::device_mesh_prof_result":
                logical_mesh.physical_mesh.prof_result,

            # Distributed compilation
            "build_option::pass_through_device_assignment": by_pass_device_assignment_check,

            # Debug options
            "auto_sharding::simplify_graph": True,
            "auto_sharding::print_strategy": False,
        }):
            compiled = xla.backend_compile(backend, xla_computation, compile_options)
        return compiled, last_s_val, last_objective

    if len(logical_mesh_choices) == 1:  # Compile with the given logical mesh
        logical_mesh = logical_mesh_choices[0]
        compiled, solution_vector, objective = _invoke_compilation(logical_mesh)
    else:  # Search for the best logical mesh
        best_logical_mesh = best_compiled = best_solution_vector = best_objective = None
        best_latency = float("inf")
        for logical_mesh in logical_mesh_choices:
            compiled, solution_vector, objective = _invoke_compilation(logical_mesh)

            if logical_mesh_search_mode == "measurement":
                strategy_config = StrategyConfig(
                    build_random_seed, logical_mesh.id_mesh.shape, solution_vector
                )
                latency = physical_mesh.profile_executable(
                    compiled, unoptimized_hlo_proto, strategy_config)
            else:
                latency = objective

            #print(logical_mesh.id_mesh.shape, objective, latency)

            if latency < best_latency:
                best_logical_mesh, best_compiled, best_solution_vector, best_objective = \
                    logical_mesh, compiled, solution_vector, objective
                best_latency = latency
        logical_mesh, compiled, solution_vector = \
            best_logical_mesh, best_compiled, best_solution_vector

    testing.last_compiled_executable = compiled
    testing.last_compiled_auto_sharding_objective = objective
    strategy_config = StrategyConfig(
        build_random_seed, logical_mesh.id_mesh.shape, solution_vector
    )
    return compiled, strategy_config


def compile_with_given_strategy(backend,
                                xla_computation,
                                num_devices,
                                by_pass_device_assignment_check,
                                strategy_config):
    """Compile an XLA computation with a given auto sharding strategy."""
    compile_options = get_compile_options(
        num_replicas=1,
        num_partitions=num_devices,
        device_assignment=np.arange(num_devices).reshape((1, -1)),
        use_spmd_partitioning=True,
        parameter_is_tupled_arguments=False,
        build_random_seed=strategy_config.build_random_seed
    )
    solution_vector = strategy_config.auto_sharding_solution_vector
    logical_mesh_shape = strategy_config.logical_mesh_shape
    with XlaPassContext({
        # Solver options
        "auto_sharding::enable": True,
        "auto_sharding::load_strategy": True,
        "auto_sharding::strategy_vector": to_int_tuple(solution_vector),

        # Device mesh
        "auto_sharding::device_mesh_ids": tuple(range(num_devices)),
        "auto_sharding::device_mesh_shape": tuple(logical_mesh_shape),

         # Distributed compilation
         "build_option::pass_through_device_assignment": by_pass_device_assignment_check,

        # Other useless but required arguments
        "auto_sharding::device_mesh_alpha": (1.0,) * len(logical_mesh_shape),
        "auto_sharding::device_mesh_beta": (1.0,) * len(logical_mesh_shape),
        "auto_sharding::device_mesh_prof_result": None,
    }):
        compiled = backend.compile(xla_computation, compile_options)
    return compiled


def _hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, logical_mesh):
    sharding_type, tile_assignment_dimensions, tile_assignment_devices, \
        _, _ = proto_tuple

    sharding = []
    mesh_mapping = []
    if sharding_type == OpSharding.Type.OTHER:
        # try to map dimension between provided mesh and real mesh
        mesh_mapping = [None] * len(logical_mesh.id_mesh.shape)
        tensor_dim_to_mesh_dim = logical_mesh.get_tensor_dim_to_mesh_dim(
            aval.shape, tile_assignment_dimensions, tile_assignment_devices)

        pt = 0
        for tensor_dim in range(len(aval.shape)):
            if tile_assignment_dimensions[tensor_dim] == 1:
                sharding.append(pxla.NoSharding())
            else:
                sharding.append(pxla.Chunked([tile_assignment_dimensions[tensor_dim]]))
                mesh_dim = tensor_dim_to_mesh_dim[tensor_dim]
                mesh_mapping[mesh_dim] = pxla.ShardedAxis(pt)
                pt += 1

        # All other dims are replicated
        for mesh_dim, _ in enumerate(mesh_mapping):
            if mesh_mapping[mesh_dim] is None:
                mesh_mapping[mesh_dim] = pxla.Replicated(logical_mesh.id_mesh.shape[mesh_dim])
    elif sharding_type == OpSharding.Type.REPLICATED:
        sharding = (pxla.NoSharding(),) * len(aval.shape)
        mesh_mapping = (pxla.Replicated(np.prod(logical_mesh.id_mesh.shape)),)
    else:
        raise NotImplementedError("Type: " + str(sharding_type))

    return pxla.ShardingSpec(sharding, mesh_mapping)


def hlo_sharding_to_sharding_spec(hlo_sharding, aval, logical_mesh_shape):
    """Convert hlo sharding to sharding spec."""
    logical_mesh = LogicalDeviceMesh(
        None, np.arange(np.prod(logical_mesh_shape)).reshape(logical_mesh_shape))
    proto_tuple = hlo_sharding.proto_tuple()
    sharding_type, _, _, tuple_shardings, _ = proto_tuple
    if sharding_type == OpSharding.Type.TUPLE:
        avals = aval
        return [_hlo_sharding_to_sharding_spec_no_tuple(shard, aval, logical_mesh)
                for (shard, aval) in zip(tuple_shardings, avals)]
    else:
        return _hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, logical_mesh)


def make_replicated_spec(aval, logical_mesh_shape):
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
def _call_solver_serialized_args(N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
                                 c_np, d_np, m_np, r_np, v_np,
                                 s_init_np=None):
    """Call the solver with serailized arguments."""
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
    for i in range(N):
        if s_follow[i] < 0:
            if s_len[i] == 1:
                s.append([1])
            else:
                num_nodes += 1
                s.append(LpVariable.matrix(f"s[{i}]",
                                           (range(s_len[i]),), cat="Binary"))
        else:
            s.append(s[s_follow[i]])

    num_edges = 0
    for (idx, (i, j)) in enumerate(E):
        if len(s[i]) == 1:
            e.append(s[j])
        elif len(s[j]) == 1:
            e.append(s[i])
        else:
            num_edges += 1
            e.append(LpVariable.matrix(f"e[{i},{j}]",
                                       (range(len(s[i]) * len(s[j])),), cat="Binary"))
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
            prob += lpSum(e[idx][row * C + col] for col in range(0, C)) <= s[i][row]

        # (g)
        for col in range(len(s[j])):
            R = len(s[i])
            C = len(s[j])
            prob += lpSum(e[idx][row * C + col] for row in range(0, R)) <= s[j][col]

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
    assert "GLPK_CMD" in pulp.listSolvers(onlyAvailable=True), \
        "Please install ILP solvers by 'sudo apt install coinor-cbc glpk-utils'"
    solver = pulp.COIN_CMD(mip=True, msg=msg, timeLimit=time_limit,
                           threads=multiprocessing.cpu_count())
    # solver = pulp.GLPK_CMD(mip=True, msg=msg, timeLimit=time_limit)
    prob.solve(solver)

    objective = float(pulp.value(prob.objective))
    status = prob.status
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
    return s_val, e_val, objective, status
