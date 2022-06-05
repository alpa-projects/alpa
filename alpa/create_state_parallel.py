from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef

from alpa.util import jaxpr_to_hlo_module


def compile_create_state_executable(fun, in_tree, out_tree_thunk, static_argnums,
                                    donated_invars, batch_invars, train_step,
                                    other_args, *avals):
    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    out_tree = out_tree_thunk()
    state_aval = tree_unflatten(out_tree, out_avals)

    executable = train_step.get_executable(state_aval, other_args)
    placement_specs = executable.get_placement_specs(0)

    name = f"{fun.__name__}_shard_parallel"
    backend = xb.get_backend("gpu")
    hlo_module = jaxpr_to_hlo_modules(name, ClosedJaxpr(jaxpr, consts),
                                      donated_invars, backend)

    return executable
