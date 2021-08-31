"""Layer clustering and remat by layer."""
from functools import wraps
from typing import List, Callable

import numba
import numpy as np

import jax
from jax import tree_flatten
from jax import lax
from jax._src.api import make_jaxpr, _check_scalar
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar, jaxpr_as_fun
from jax.interpreters import xla

from .primitive_def import mark_pipeline

# TODO: different operations takes different time
# e.g. add v.s. pow

gpu_backend = xc.get_local_backend("gpu")


def call_to_xla_computation(eqn: JaxprEqn):
    xe = xc._xla
    prim = eqn.primitive
    backend = gpu_backend

    c = xb.make_computation_builder(f"primitive_computation_{prim.name}")

    name = xla.extend_name_stack(prim.name)

    op_metadata = xla.make_op_metadata(prim, eqn.params)
    c.set_op_metadata(op_metadata)
    xla_args, _ = xla._xla_callable_args(
        c, list(map(lambda x: x.aval, eqn.invars)),
        len(eqn.invars) > 100)
    axis_env = xla.AxisEnv(1, (), ())

    new_params = xla.check_backend_params(eqn.params, backend)
    rule = xla.call_translations[eqn.primitive]
    ans = rule(c, axis_env, xla_args, name, backend=backend, **new_params)

    assert isinstance(ans, xe.XlaOp)
    c.clear_op_metadata()
    try:
        return c.build(ans)
    except RuntimeError as e:
        msg = (
            " ".join(map(str, e.args)) + "\n"
            "This is a bug in JAX's shape-checking rules; please report it!\n"
            "https://github.com/google/jax/issues\n")
        raise RuntimeError(msg) from e


def eqn_flops(eqn: JaxprEqn) -> float:
    if eqn.primitive in xla.call_translations:
        xla_computation = call_to_xla_computation(eqn)
    else:
        xla_computation = xla.primitive_subcomputation(
            eqn.primitive, *map(lambda x: x.aval, eqn.invars), **eqn.params)
    hlo_module = xla_computation.as_hlo_module()
    properties = xc._xla.hlo_module_cost_analysis(gpu_backend, hlo_module)
    return properties["flops"] if "flops" in properties else 0.0


def cluster_edges_cost(start: List['JaxprEqn'], end: List['JaxprEqn']):
    out_tensors = set()
    for eqn in start:
        out_tensors = out_tensors.union(set(eqn.outvars))
    in_tensors = set()
    for eqn in end:
        for invar in eqn.invars:
            if isinstance(invar, Var) and invar in out_tensors:
                in_tensors.add(invar)
    acc = 0
    for in_tensor in in_tensors:
        acc += in_tensor.aval.size * in_tensor.aval.dtype.itemsize
    return acc


non_trivial_primitive = [lax.dot_general_p, lax.conv_general_dilated_p]


def slice_jaxpr(jaxpr: Jaxpr, layer_num: int, eps: float):
    length = len(jaxpr.eqns)
    non_trivial = [eqn.primitive in non_trivial_primitive for eqn in jaxpr.eqns]
    non_trivial = np.array(non_trivial)
    C = np.full((length + 1, length + 1), 0, dtype=np.float32)
    # init

    outvars = set()
    for k in range(0, length + 1):
        if k > 0:
            outvars = outvars.union(jaxpr.eqns[k - 1].outvars)
        invars = set()
        tot = 0
        for r in range(k + 1, length + 1):
            for invar in jaxpr.eqns[r - 1].invars:
                if isinstance(invar, Var) and invar in outvars\
                  and invar not in invars:
                    invars.add(invar)
                    tot += invar.aval.size
            C[k, r] = tot

    LAYER_HEAVY_OP_BOUND = non_trivial.sum() / layer_num
    LAYER_HEAVY_OP_BOUND = max(LAYER_HEAVY_OP_BOUND + 1,
                               LAYER_HEAVY_OP_BOUND * (1 + eps))

    @numba.jit(nopython=True)
    def DP(C):
        A = np.full((length + 1, layer_num + 1), np.inf, dtype=np.float32)
        A_argmin = np.full((length + 1, layer_num + 1), -1, dtype=np.int32)
        B = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
        A[0, 0] = 0
        for l in range(1, length + 1):
            cnt = 0
            for r in range(l, length + 1):
                if non_trivial[r - 1]:
                    cnt += 1
                if cnt < 1:
                    continue
                elif cnt <= LAYER_HEAVY_OP_BOUND:
                    B[l, r] = 0
                else:
                    break
        for q in range(1, layer_num + 1):
            for r in range(1, length + 1):
                for k in range(0, r):
                    new_value = A[k, q - 1] + B[k + 1, r] + C[k, r]
                    if new_value < A[r, q]:
                        A[r, q] = new_value
                        A_argmin[r, q] = k
        return A_argmin

    A_argmin = DP(C)

    reversed_sliced_eqns = []

    r = length
    for q in range(layer_num, 0, -1):
        k = A_argmin[r, q]
        reversed_sliced_eqns.append(jaxpr.eqns[k:r])
        r = k
    assert r == 0, 'no solution for layer clustering' if r == -1 else 'unknown error'
    return list(reversed(reversed_sliced_eqns))


def reconstruct_by_solution(closed_jaxpr: ClosedJaxpr,
                            sliced_eqns) -> List[ClosedJaxpr]:
    N = len(sliced_eqns)
    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_consts = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    global_outvars = set(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    result = []
    layer_invars = [set() for _ in range(N)]
    layer_outvars = [set() for _ in range(N)]
    layer_consts = [dict() for _ in range(N)]
    var_layer_dict = {}
    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if isinstance(var, Literal):
                    continue
                if var in global_consts:
                    layer_consts[i][var] = global_consts[var]
                elif var in global_invars:
                    layer_invars[i].add(var)
                elif var_layer_dict[var] != i:
                    layer_invars[i].add(var)
                    layer_outvars[var_layer_dict[var]].add(var)
                else:
                    assert var_layer_dict[var] == i
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i
                if var in global_outvars:
                    layer_outvars[i].add(var)
    for i, eqns in enumerate(sliced_eqns):
        new_jaxpr = Jaxpr(list(layer_consts[i].keys()), list(layer_invars[i]),
                          list(layer_outvars[i]), eqns)
        new_closed_jaxpr = ClosedJaxpr(new_jaxpr,
                                       list(layer_consts[i].values()))
        result.append(new_closed_jaxpr)
    return result


def forward(fn: Callable,
            layer_num: int,
            eps: float = 0,
            use_pipeline: bool = False,
            use_remat: bool = False):
    ''''''
    if use_remat or use_pipeline:

        @wraps(fn)
        def wrapped(*args):
            origin_jaxpr = make_jaxpr(fn)(*args)
            solution = slice_jaxpr(origin_jaxpr, layer_num, eps)
            global_invars = origin_jaxpr.jaxpr.invars

            sliced_jaxprs = reconstruct_by_solution(origin_jaxpr, solution)
            sliced_callables = [
                jax.remat(jaxpr_as_fun(layer))
                if use_remat else jaxpr_as_fun(layer) for layer in sliced_jaxprs
            ]

            flatten_inputs, _ = tree_flatten(args)
            glob_vars = dict(zip(global_invars, flatten_inputs))
            cnt = 0
            for (closed_jaxpr, runnable) in zip(sliced_jaxprs,
                                                sliced_callables):
                args = []
                for invar in closed_jaxpr.jaxpr.invars:
                    args.append(glob_vars[invar])
                if use_pipeline:
                    args = mark_pipeline(*args,
                                         name=str(cnt),
                                         mark_type='start')
                ans = runnable(*args)
                if use_pipeline:
                    ans = mark_pipeline(*ans, name=str(cnt), mark_type='end')
                for i, outvar in enumerate(closed_jaxpr.jaxpr.outvars):
                    glob_vars[outvar] = ans[i]
                cnt += 1
            assert len(ans) == 1
            _check_scalar(ans[0])
            return ans[0]

        return wrapped
    else:
        return fn
