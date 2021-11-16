"""Cluster small operators into layers.
Do rematerialization at the boundary of layer."""

from functools import partial, wraps
import logging
from typing import List, Callable

from jax._src.tree_util import tree_unflatten
from jax import tree_flatten
from jax import lax
from jax._src.api import make_jaxpr
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Jaxpr, Var, jaxpr_as_fun, CallPrimitive
from jax.interpreters import xla
import numba
import numpy as np

from parax.util import get_cross_slice_vars
from parax.pipeline_parallel.manual_layer_slicing import insert_marker, manual_layer_slicing, remat_jaxpr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

gpu_backend = xc.get_local_backend("gpu")


def call_to_xla_computation(eqn: JaxprEqn):
    """Convert a jaxpr equation to a XLA computation for FLOP analysis."""
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
    """Get the FLOP of a jaxpr equation."""
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


def heavy_count(eqn):
    if eqn.primitive in non_trivial_primitive:
        return 1
    if isinstance(eqn.primitive, CallPrimitive):
        assert "call_jaxpr" in eqn.params
        called = eqn.params["call_jaxpr"]
        cnt = 0
        for subjaxpr_eqn in called.eqns:
            cnt += heavy_count(subjaxpr_eqn)
        return cnt
    return 0


def is_nontrivial(eqn):
    return heavy_count(eqn) > 0


def get_stat(jaxpr):
    length = len(jaxpr.eqns)
    non_trivial = [is_nontrivial(eqn) for eqn in jaxpr.eqns]
    non_trivial = np.array(non_trivial, dtype=np.int32)
    Cost = np.full((length + 1, length + 1), 0, dtype=np.float32)
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
            Cost[k, r] = tot
    return non_trivial, Cost


def slice_jaxpr(jaxpr: Jaxpr,
                layer_num: int,
                eps: float,
                return_value=False,
                stat=None,
                cost_criteria="flops"):
    layer_num = int(layer_num)
    length = len(jaxpr.eqns)
    if stat:
        non_trivial, Cost = stat
    else:
        non_trivial, Cost = get_stat(jaxpr)

    if cost_criteria == "flops":
        cost_fn = eqn_flops
    elif cost_criteria == "count":
        cost_fn = heavy_count
    else:
        raise ValueError(f"Unrecoginzed cost criteria {cost_criteria}")
    flops = [
        cost_fn(eqn) if is_nontrivial else 0
        for is_nontrivial, eqn in zip(non_trivial, jaxpr.eqns)
    ]
    flops = np.array(flops, dtype=np.float64)
    flops_avg = flops.sum() / layer_num
    flops_bound = flops_avg * (1 + eps)
    if cost_criteria == "count":
        flops_bound = max(flops_bound, flops_avg + 5)
    LAYER_HEAVY_OP_LOW_BOUND = 3
    if sum(non_trivial) / layer_num < LAYER_HEAVY_OP_LOW_BOUND:
        LAYER_HEAVY_OP_LOW_BOUND = int(sum(non_trivial) / layer_num)
        logger.warning(
            "Too few non-trivial ops (dot, conv), which may influence auto-sharding performance"
        )

    @numba.jit(nopython=True)
    def Init():
        Blocked = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
        for l in range(1, length + 1):
            cnt = 0
            flops_cnt = 0
            for r in range(l, length + 1):
                if non_trivial[r - 1]:
                    cnt += 1
                    flops_cnt += flops[r - 1]
                if cnt < LAYER_HEAVY_OP_LOW_BOUND:
                    if flops_cnt >= flops_bound:
                        Blocked[l, r] = 0
                    continue
                if (flops_cnt >= flops_bound and non_trivial[r - 1] and
                        cnt > LAYER_HEAVY_OP_LOW_BOUND):
                    break
                Blocked[l, r] = 0
        return Blocked

    @numba.jit(nopython=True)
    def DP(Cost, Blocked):
        MaxCost = np.full((length + 1, layer_num + 1), np.inf, dtype=np.float32)
        SumCostUnderMax = np.full((length + 1, layer_num + 1),
                                  np.inf,
                                  dtype=np.float32)
        MaxCost_argmin = np.full((length + 1, layer_num + 1),
                                 -1,
                                 dtype=np.int32)
        MaxCost[0, 0] = 0
        SumCostUnderMax[0, 0] = 0
        for q in range(1, layer_num + 1):
            for r in range(1, length + 1):
                for k in range(0, r):
                    new_value = max(MaxCost[k, q - 1],
                                    Blocked[k + 1, r] + Cost[k, r])
                    new_sum = (SumCostUnderMax[k, q - 1] + Blocked[k + 1, r] +
                               Cost[k, r])
                    if (new_value < MaxCost[r, q] or
                        (new_value <= MaxCost[r, q] *
                         (1 + 1e-4) and new_sum < SumCostUnderMax[r, q])):
                        MaxCost[r, q] = new_value
                        SumCostUnderMax[r, q] = new_sum
                        MaxCost_argmin[r, q] = k
        return MaxCost_argmin, MaxCost[length, layer_num]

    Blocked = Init()
    A_argmin, value = DP(Cost, Blocked)

    reversed_sliced_eqns = []

    r = length
    for q in range(layer_num, 0, -1):
        k = A_argmin[r, q]
        reversed_sliced_eqns.append(jaxpr.eqns[k:r])
        r = k
    assert r == 0, "no solution for layer clustering" if r == -1 else "unknown error"
    solution = list(reversed(reversed_sliced_eqns))
    if return_value:
        return solution, value
    return solution


def search_layer_num(jaxpr, eps, layer_eps=0):
    stat = non_trivial, Cost = get_stat(jaxpr)
    l = 2
    r = int(non_trivial.sum() / 3) + 1
    _, l_val = slice_jaxpr(jaxpr, l, eps, True, stat)
    while r - l > 1:
        mid = int((l + r) / 2)
        _, mid_val = slice_jaxpr(jaxpr, mid, eps, True, stat)
        if mid_val > l_val * (1 + layer_eps):
            r = mid
        else:
            l = mid
    return l


def automatic_layer_slicing(fn: Callable,
                            layer_num: int,
                            eps: float = 0.6,
                            cost_criteria: str = "flops",
                            use_pipeline: bool = False,
                            use_remat: bool = False,
                            layer_eps: float = 0):
    """
    Automatically slice the jaxpr into layers.
    Pipeline markers and rematerialization can be added at the boundary of layers.

    Args:
        fun: The forward function
        layer_num: The number of output layers. Use binary search if value is "auto"
        eps: The imbalance tolerance among layers.
        cost_criteria: If "flops", use FLOPs of each eqn for rough computation balance;
            If "count", simply count number of dot/conv.
        use_pipeline: Whether to insert pipeline markers at the boundary of layers.
        use_remat: Whether to use rematerialization at the boundary of layers.
        layer_eps: The imbalance tolerance for binary search of layer_num
    """
    if use_remat or use_pipeline:

        def get_sliced(*args):
            origin_jaxpr, out_shape_tree = make_jaxpr(fn,
                                                      static_argnums=(),
                                                      return_shape=True)(*args)
            nonlocal layer_num
            if layer_num == "auto":
                layer_num = search_layer_num(origin_jaxpr, eps, layer_eps)

            slices = slice_jaxpr(origin_jaxpr,
                                 layer_num,
                                 eps,
                                 cost_criteria=cost_criteria)
            print("-" * 20, "Automatic layer slicing stats", "-" * 20)
            print(f"layer_num: {layer_num}")
            print(" - Number of Jaxpr eqns in each stage:")
            for i, slice in enumerate(slices):
                print(f"Layer {i}: {len(slice)}")
            print(" - Invars of each stage:")
            get_cross_slice_vars(origin_jaxpr.jaxpr, slices)
            print("-" * 70)
            return origin_jaxpr, slices, out_shape_tree

        @wraps(fn)
        @manual_layer_slicing
        def wrapped(*args):
            origin_jaxpr, slices, out_shape_tree = get_sliced(*args)
            transformation = partial(
                remat_jaxpr,
                use_pipeline=use_pipeline) if use_remat else insert_marker
            new_jaxpr = transformation(origin_jaxpr, slices)

            flatten_args, _ = tree_flatten(args)

            ans = jaxpr_as_fun(new_jaxpr)(*flatten_args)
            _, out_tree = tree_flatten(out_shape_tree)
            return tree_unflatten(out_tree, ans)

        wrapped.get_sliced = get_sliced
        return wrapped
    else:
        return fn
