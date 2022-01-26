"""Functions related with computing the stats during layer construction."""
from typing import List, Set

from jax import lax
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Var, CallPrimitive, DropVar, Literal
from jax.interpreters import xla
from alpa.util import OrderedSet


def call_to_xla_computation(eqn: JaxprEqn):
    """Convert a jaxpr equation to a XLA computation for FLOP analysis.
    
    Reference code: jax/jax/interpreters/xla.py::jaxpr_subcomp
    """
    xe = xc._xla
    prim = eqn.primitive
    backend = xb.get_backend("gpu")

    c = xc.XlaBuilder(f"primitive_computation_{prim.name}")
    name_stack = xla.extend_name_stack(prim.name)

    def aval(v):
        if type(v) is Literal:
            return abstractify(v.val)
        else:
            return v.aval

    op_metadata = xla.make_op_metadata(prim,
                                       eqn.params,
                                       source_info=eqn.source_info)
    c.set_op_metadata(op_metadata)
    in_nodes, _ = xla._xla_callable_args(
        c, list(map(lambda x: x.aval, eqn.invars)),
        len(eqn.invars) > 100)
    axis_env = xla.AxisEnv(1, (), ())
    ctx = xla.TranslationContext(c, backend.platform, axis_env, name_stack)
    rule = xla._translations[eqn.primitive]
    ans = rule(ctx, map(aval, eqn.invars), map(aval, eqn.outvars), *in_nodes,
               **eqn.params)
    c.clear_op_metadata()

    try:
        ans = xc.ops.Tuple(c, ans)
        return c.build(ans)
    except RuntimeError as e:
        msg = (
            " ".join(map(str, e.args)) + "\n"
            "This is a bug in JAX's shape-checking rules; please report it!\n"
            "https://github.com/google/jax/issues\n")
        raise RuntimeError(msg) from e


def eqn_flops(eqn: JaxprEqn) -> float:
    """Get the FLOP of a jaxpr equation."""
    if eqn.primitive in xla._translations:
        xla_computation = call_to_xla_computation(eqn)
    else:
        xla_computation = xla.primitive_subcomputation(
            eqn.primitive, *map(lambda x: x.aval, eqn.invars), **eqn.params)
    hlo_module = xla_computation.as_hlo_module()
    properties = xc._xla.hlo_module_cost_analysis(xb.get_backend("gpu"),
                                                  hlo_module)
    return properties["flops"] if "flops" in properties else 0.0


def cluster_edges_cost(start: List['JaxprEqn'], end: List['JaxprEqn']):
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
        print(f'Layer {i} has inputs:')
        for invar in invars:
            print(invar, invar.aval.shape, 'from layer', defined[invar])


def log_layer_slicing_stats(origin_jaxpr, slices):
    """Print the layer slicing stats."""
    stage_flops = []
    stage_heavy_ops = []
    for eqns in slices:
        stage_flops.append(sum([eqn_flops(eqn) for eqn in eqns]))
        stage_heavy_ops.append(sum([heavy_count(eqn) for eqn in eqns]))

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
    input_vars = set([v for v in eqn.invars if isinstance(v, Var)])
    size = sum([(var.aval.size * var.aval.dtype.itemsize)
                for var in invars.intersection(input_vars)])
    return size
