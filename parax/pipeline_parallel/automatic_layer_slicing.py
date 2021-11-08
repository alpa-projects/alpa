"""Cluster small operators into layers.
Do rematerialization at the boundary of layer."""

from functools import partial, wraps
from typing import List, Callable

from jax._src.tree_util import tree_unflatten
import jax
from jax import tree_flatten
from jax import lax
from jax._src.api import make_jaxpr, _check_scalar
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Jaxpr, Var, jaxpr_as_fun
from jax.interpreters import xla
import numba
import numpy as np

from parax.pipeline_parallel.manual_layer_slicing import insert_marker, manual_layer_slicing, remat_jaxpr

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
    assert r == 0, "no solution for layer clustering" if r == -1 else "unknown error"
    return list(reversed(reversed_sliced_eqns))


def automatic_layer_slicing(fn: Callable,
                            layer_num: int,
                            eps: float = 0,
                            use_pipeline: bool = False,
                            use_remat: bool = False):
    """
    Automatically slice the jaxpr into layers.
    Pipeline markers and rematerialization can be added at the boundary of layers.

    Args:
        fun: The forward function
        layer_num: The number of output layers
        eps: A parameter to control the imbalance tolerance among layers.
        use_pipeline: Whether to insert pipeline markers at the boundary of layers.
        use_remat: Whether to use rematerialization at the boundary of layers.
    """
    if use_remat or use_pipeline:

        @wraps(fn)
        @manual_layer_slicing
        def wrapped(*args):
            origin_jaxpr, out_shape_tree = make_jaxpr(fn,
                                                      static_argnums=(),
                                                      return_shape=True)(*args)
            flatten_args, _ = tree_flatten(args)

            slices = slice_jaxpr(origin_jaxpr, layer_num, eps)
            transformation = partial(
                remat_jaxpr,
                use_pipeline=use_pipeline) if use_remat else insert_marker
            new_jaxpr = transformation(origin_jaxpr, slices)
            ans = jaxpr_as_fun(new_jaxpr)(*flatten_args)
            _, out_tree = tree_flatten(out_shape_tree)
            return tree_unflatten(out_tree, ans)

        return wrapped
    else:
        return fn
