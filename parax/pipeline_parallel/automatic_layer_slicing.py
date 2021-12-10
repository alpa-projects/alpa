"""Cluster small operators into layers.
Do rematerialization at the boundary of layer."""

from functools import partial, wraps
import logging
from typing import Callable

from jax._src.tree_util import tree_unflatten
from jax import tree_flatten
from jax._src.api import make_jaxpr
from jax.core import Jaxpr, Var, jaxpr_as_fun
import numba
import numpy as np

from parax.util import OrderedSet
from parax.pipeline_parallel.manual_layer_slicing import (insert_marker,
                                                          manual_layer_slicing,
                                                          remat_jaxpr)
from parax.pipeline_parallel.layer_stats import (is_nontrivial, eqn_flops, heavy_count, log_layer_slicing_stats)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_stat(jaxpr):
    length = len(jaxpr.eqns)
    non_trivial = [is_nontrivial(eqn) for eqn in jaxpr.eqns]
    non_trivial = np.array(non_trivial, dtype=np.int32)
    Cost = np.full((length + 1, length + 1), 0, dtype=np.float32)
    # init

    outvars = OrderedSet()
    for k in range(0, length + 1):
        if k > 0:
            outvars = outvars.union(jaxpr.eqns[k - 1].outvars)
        invars = OrderedSet()
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
        SolutionImbalance = np.full((length + 1, layer_num + 1),
                                    np.inf,
                                    dtype=np.float32)
        MaxCost[0, 0] = 0
        SumCostUnderMax[0, 0] = 0
        # Currently use variance to measure imbalance
        for r in range(0, length + 1):
            SolutionImbalance[r, 0] = 0

        for q in range(1, layer_num + 1):
            for r in range(1, length + 1):
                for k in range(0, r):
                    new_value = max(MaxCost[k, q - 1],
                                    Blocked[k + 1, r] + Cost[k, r])
                    new_sum = (SumCostUnderMax[k, q - 1] + Blocked[k + 1, r] +
                               Cost[k, r])
                    new_imbalance = (SolutionImbalance[k, q - 1] + k**2 / q -
                                     r**2 / (q + 1) + (r - k)**2)
                    if (new_value < MaxCost[r, q] or
                        (new_value <= MaxCost[r, q] * (1 + 1e-4) and
                         (new_sum < SumCostUnderMax[r, q] or
                          (new_sum <= SumCostUnderMax[r, q] * (1 + 1e-4) and
                           new_imbalance < SolutionImbalance[r, q])))):
                        MaxCost[r, q] = new_value
                        SumCostUnderMax[r, q] = new_sum
                        MaxCost_argmin[r, q] = k
                        SolutionImbalance[r, q] = new_imbalance
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

    solution_info = {
        "total_cost": value,
    }
    return solution, solution_info


def search_layer_num(jaxpr, eps, layer_eps=0):
    stat = non_trivial, Cost = get_stat(jaxpr)
    l = 2
    r = int(non_trivial.sum() / 3) + 1
    _, solution_info = slice_jaxpr(jaxpr, l, eps, stat)
    l_val = solution_info["total_cost"]
    while r - l > 1:
        mid = int((l + r) / 2)
        _, solution_info = slice_jaxpr(jaxpr, mid, eps, stat)
        mid_val = solution_info["total_cost"]
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

            slices, solution_info = slice_jaxpr(origin_jaxpr,
                                                layer_num,
                                                eps,
                                                cost_criteria=cost_criteria)
            log_layer_slicing_stats(origin_jaxpr, slices)
            return origin_jaxpr, slices, out_shape_tree


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

        if use_pipeline:
            wrapped = manual_layer_slicing(wrapped)
        wrapped = wraps(fn)(wrapped)

        wrapped.get_sliced = get_sliced
        return wrapped
    else:
        return fn
