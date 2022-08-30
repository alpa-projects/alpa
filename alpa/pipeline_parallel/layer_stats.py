"""Functions related with computing the stats during layer construction."""
from typing import List, Set

from jax import lax
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Var, CallPrimitive, DropVar, Literal, Jaxpr, ClosedJaxpr
from alpa.util import OrderedSet, jaxpr_to_hlo_module


def eqn_flops(eqn: JaxprEqn) -> float:
    """Get the FLOP of a jaxpr equation."""
    jaxpr = Jaxpr([], eqn.invars, eqn.outvars, [eqn])
    closed_jaxpr = ClosedJaxpr(jaxpr, [])
    backend = xb.get_backend("gpu")
    if any(isinstance(x, Literal) for x in eqn.invars):
        # A temporary workaround
        return 0
    hlo_module = jaxpr_to_hlo_module("tmp", closed_jaxpr, [
        False,
    ] * len(eqn.invars), backend)
    properties = xc._xla.hlo_module_cost_analysis(  # pylint: disable=protected-access
        backend, hlo_module)
    return properties["flops"] if "flops" in properties else 0.0


def cluster_edges_cost(start: List["JaxprEqn"], end: List["JaxprEqn"]):
    """Calculates the cost of cluster edges."""
    out_tensors = OrderedSet()
    for eqn in start:
        out_tensors = out_tensors.union(OrderedSet(eqn.outvars))
    in_tensors = OrderedSet()
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
    """Check the number of heavy ops in the eqn."""
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
    """Check if the eqn is nontrivial."""
    return heavy_count(eqn) > 0


def get_cross_slice_vars(jaxpr, slices):
    """TODO(zhuohan):doscstring."""
    defined = {}
    stage_invars = [OrderedSet() for _ in slices]
    for invar in jaxpr.invars:
        defined[invar] = -1
    for invar in jaxpr.constvars:
        defined[invar] = -1
    for i, sliced in enumerate(slices):
        for eqn in sliced:
            for outvar in eqn.outvars:
                if isinstance(outvar, DropVar):
                    continue
                defined[outvar] = i
    for i, sliced in enumerate(slices):
        for eqn in sliced:
            for invar in eqn.invars:
                if not isinstance(invar, Var):
                    continue
                if defined[invar] >= 0 and defined[invar] != i:
                    stage_invars[i].add(invar)
    for i, invars in enumerate(stage_invars):
        print(f"Layer {i} has inputs:")
        for invar in invars:
            print(invar, invar.aval.shape, "from layer", defined[invar])


def log_layer_slicing_stats(origin_jaxpr, slices):
    """Print the layer slicing stats."""
    stage_flops = []
    stage_heavy_ops = []
    for eqns in slices:
        stage_flops.append(sum(eqn_flops(eqn) for eqn in eqns))
        stage_heavy_ops.append(sum(heavy_count(eqn) for eqn in eqns))

    print("-" * 20, "Layer slicing stats", "-" * 20)
    print(f"layer_num: {len(slices)}")
    print(" - Number of Jaxpr eqns in each stage:")
    for i, s in enumerate(slices):
        print(f"Layer {i}: #eqns={len(s)},"
              f" flop={stage_flops[i] / (1000 ** 4):.3f} TFlop,"
              f" #heavy_ops={stage_heavy_ops[i]}")
    print(" - Invars of each stage:")
    get_cross_slice_vars(origin_jaxpr.jaxpr, slices)
    print("-" * 61)


def global_invar_size(invars: Set[Var], eqn: JaxprEqn):
    input_vars = {v for v in eqn.invars if isinstance(v, Var)}
    size = sum((var.aval.size * var.aval.dtype.itemsize)
               for var in invars.intersection(input_vars))
    return size
