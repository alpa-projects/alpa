"""Use the auto sharding pass in XLA"""
from functools import partial
import multiprocessing
import pickle
import sys
import time
import traceback
from warnings import warn


import numpy as np

import jax
from jax import linear_util as lu
from jax.interpreters import xla, pxla, partial_eval as pe
from jax.lib import xla_bridge as xb
from jax.lib import xla_client as xc
from jax._src.util import (partial, unzip2, unzip3, prod, safe_map, safe_zip,
                           extend_name_stack, wrap_name, assert_unreachable,
                           tuple_insert, tuple_delete, curry)
from jaxlib.xla_client import OpSharding

from parax import testing

xops = xc.ops

def auto_sharding_callable(
    fun: lu.WrappedFun,
    out_tree_thunk,
    devices,
    donated_invars,
    memory_budget_per_device,
    *avals
):
    devices = devices or np.array(jax.devices())

    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)

    tuple_args = len(avals) > 100  # pass long arg lists as tuple for TPU

    # Make xla arguments
    c = xb.make_computation_builder(f"auto_shard_{fun.__name__}")
    xla_consts = map(partial(xb.constant, c), consts)
    xla_args, donated_invars = xla._xla_callable_args(c, avals, tuple_args, donated_invars=donated_invars)

    # Convert jaxpr to XLA HLO
    backend_name = 'gpu'
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
    transformed_name = fun.__name__
    out_nodes = xla.jaxpr_subcomp(
        c, jaxpr, backend_name, axis_env, xla_consts,
        extend_name_stack(wrap_name(transformed_name, 'auto_sharding')), *xla_args)
    out_tuple = xops.Tuple(c, out_nodes)

    # Set up aliases (donating invars)
    backend = xb.get_backend(backend_name)
    if backend.platform in ("gpu", "tpu"):
        donated_invars = xla.set_up_aliases(c, xla_args, out_tuple, donated_invars, tuple_args)
    if any(donated_invars):
        # TODO(tomhennigan): At call time we should mark these buffers as deleted.
        unused_donations = [str(c.GetShape(a))
                            for a, d in zip(xla_args, donated_invars) if d]
        warn("Some donated buffers were not usable: {}".format(", ".join(unused_donations)))

    # Compile
    device_ids = np.array([x.id for x in devices])
    num_replicas = 1
    num_partitions = len(device_ids)
    device_assignment = device_ids.reshape((num_replicas, num_partitions))
    spmd_lowering = True
    compile_options = xb.get_compile_options(
        num_replicas=num_replicas,
        num_partitions=num_partitions,
        device_assignment=device_assignment,
        use_spmd_partitioning=spmd_lowering,
    )
    executable_build_options = compile_options.executable_build_options
    executable_build_options.use_auto_sharding = True
    if memory_budget_per_device:
        executable_build_options.memory_budget_per_device = int(memory_budget_per_device)
    compile_options.parameter_is_tupled_arguments = tuple_args
    built = c.Build(out_tuple)
    compiled = xla.backend_compile(backend, built, compile_options)

    testing.last_compiled_executable = compiled

    # Handle args (re-shard if the layout is not the same)
    input_shardings = compiled.hlo_modules()[0].spmd_parameters_shardings()
    input_sharding_specs = [hlo_sharding_to_sharding_spec(proto_tuple, aval, num_partitions)
                           for (proto_tuple, aval) in zip(input_shardings, avals)]
    input_indices = [pxla.spec_to_indices(aval.shape, spec) for
                     aval, spec in zip(avals, input_sharding_specs)]
    handle_args = partial(pxla.shard_args, compiled.local_devices(), input_indices)

    # Handle output
    output_sharding = compiled.hlo_modules()[0].spmd_output_sharding()
    output_sharding_specs = hlo_sharding_to_sharding_spec(output_sharding, out_avals, num_partitions)
    handle_outs = pxla.avals_to_results_handler(num_replicas, num_partitions,
                                                output_sharding_specs, out_avals)

    return partial(pxla.execute_replicated, compiled, backend, handle_args, handle_outs)


def hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, num_partitions):
    sharding_type, tile_assignment_dimensions, tile_assignment_devices,\
        tuple_shardings, replicate_on_last_tile_dim = proto_tuple

    sharding = []
    mesh_mapping = []
    if sharding_type == OpSharding.Type.OTHER:
        for i in range(len(tile_assignment_dimensions)):
            sharding.append(pxla.Chunked([tile_assignment_dimensions[i]]))
            mesh_mapping.append(pxla.ShardedAxis(i))
    elif sharding_type == OpSharding.Type.REPLICATED:
        sharding = (pxla.NoSharding(),) * len(aval.shape)
        mesh_mapping = (pxla.Replicated(num_partitions),)
    else:
        raise NotImplementedError("Type: " + str(sharding_type))

    return pxla.ShardingSpec(sharding, mesh_mapping)


def hlo_sharding_to_sharding_spec(hlo_sharding, aval, num_partitions):
    proto_tuple = hlo_sharding.proto_tuple()
    sharding_type, tile_assignment_dimensions, tile_assignment_devices,\
        tuple_shardings, replicate_on_last_tile_dim = proto_tuple
    if sharding_type == OpSharding.Type.TUPLE:
        avals = aval
        return [hlo_sharding_to_sharding_spec_no_tuple(shard, aval, num_partitions)
                for (shard, aval) in zip(tuple_shardings, avals)]
    else:
        return hlo_sharding_to_sharding_spec_no_tuple(proto_tuple, aval, num_partitions)


def call_solver_serialized_args(*args):
    try:
        ret = _call_solver_serialized_args(*args)
    except AssertionError:
        ret = None
        info = str(traceback.format_exc()[:-1])
    except Exception:
        ret = None
        info = str(traceback.format_exc()[:-1])

    if ret is None:
        print(info)

    return ret


def _call_solver_serialized_args(N, M, s_len_np, s_follow_np, E_np, A_np, L_np,
                                 c_np, d_np, m_np, r_np, v_np,
                                 s_init_np=None):
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
        """Get the index of non-zero item in a vector"""
        ct = 0
        ret = None
        for i in range(len(binary_vector)):
            if pulp.value(binary_vector[i]):
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

    for i in range(N):
        if s_follow[i] < 0:
            if s_len[i] == 1:
                s.append([1])
            else:
                s.append(LpVariable.matrix(f"s[{i}]",
                    (range(s_len[i]),), cat="Binary"))
        else:
            s.append(s[s_follow[i]])

    for (idx, (i, j)) in enumerate(E):
        if len(s[i]) == 1:
            e.append(s[j])
        elif len(s[j]) == 1:
            e.append(s[i])
        else:
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
        # todo: set edge value

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
        if (i,j) in alias_set:
            raise ValueError(f"Duplicated edges: {(i, j)}")
        else:
            alias_set.add((i, j))
            alias_set.add((j, i))

        for row in range(len(s[i])):
            for col in range(len(s[j])):
                if v[idx][row * C + col] > 0.5:
                    prob += s[i][row] + s[j][col] <= 1

    msg = False
    time_limit = 2000
    assert "GLPK_CMD" in pulp.listSolvers(onlyAvailable=True), \
        "Please install ILP solvers by 'sudo apt install coinor-cbc glpk-utils'"
    #solver = pulp.COIN_CMD(mip=True, msg=msg, timeLimit=time_limit,
    #                       threads=multiprocessing.cpu_count())
    solver = pulp.GLPK_CMD(mip=True, msg=msg, timeLimit=time_limit)
    result = prob.solve(solver)

    verbose = False
    objective = float(pulp.value(prob.objective))
    status = prob.status
    if verbose:
        print("Auto-sharding ILP status:", LpStatus[status])
        print("Auto-sharding ILP value:", objective)
        print("Auto-sharding ILP graph nodes:", np.sum(s_follow_np < 0))
        print(f"Auto-sharding ILP Time: {time.time() - tic:.2f}")

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

    testing.last_compiled_auto_sharding_objective = objective
    return s_val, e_val, objective, status

