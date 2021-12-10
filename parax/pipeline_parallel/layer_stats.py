from typing import List
from jax import lax
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import JaxprEqn, Var, CallPrimitive
from jax.interpreters import xla
from parax.util import get_cross_slice_vars, OrderedSet


def call_to_xla_computation(eqn: JaxprEqn):
    """Convert a jaxpr equation to a XLA computation for FLOP analysis."""
    xe = xc._xla
    prim = eqn.primitive
    backend = xc.get_local_backend("gpu")

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
    properties = xc._xla.hlo_module_cost_analysis(xc.get_local_backend("gpu"),
                                                  hlo_module)
    return properties["flops"] if "flops" in properties else 0.0


def cluster_edges_cost(start: List['JaxprEqn'], end: List['JaxprEqn']):
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


def log_layer_slicing_stats(origin_jaxpr, slices):
    stage_flops = []
    stage_heavy_ops = []
    for eqns in slices:
        stage_flops.append(sum([eqn_flops(eqn) for eqn in eqns]))
        stage_heavy_ops.append(sum([heavy_count(eqn) for eqn in eqns]))

    print("-" * 20, "Layer slicing stats", "-" * 20)
    print(f"layer_num: {len(slices)}")
    print(" - Number of Jaxpr eqns in each stage:")
    for i, slice in enumerate(slices):
        print(f"Layer {i}: #eqns={len(slice)},"
              f" flop={stage_flops[i] / (1000 ** 4):.3f} TFlop,"
              f" #heavy_ops={stage_heavy_ops[i]}")
    print(" - Invars of each stage:")
    get_cross_slice_vars(origin_jaxpr.jaxpr, slices)
    print("-" * 70)
