"""Compile executables for creating training state distributedly."""
from jax._src.lib import xla_bridge as xb, xla_extension as xe
from jax.core import (Jaxpr, ClosedJaxpr, Literal, new_jaxpr_eqn, gensym,
                      get_aval, raise_to_shaped, AbstractValue)
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef

from alpa.shard_parallel.auto_sharding import run_spmd_partitioner_pass
from alpa.mesh_executable import NormalMeshDriverExecutable
from alpa.measure_record import StrategyConfig
from alpa.util import jaxpr_to_hlo_module


def compile_create_state_executable(fun, in_tree, out_tree_thunk, static_argnums,
                                    donated_invars, batch_invars, train_step,
                                    other_args, *avals):
    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    out_tree = out_tree_thunk()
    state_aval = tree_unflatten(out_tree, out_avals)

    executable = train_step.get_executable(state_aval, other_args)
    placement_specs = executable.get_placement_specs()[0]

    name = f"{fun.__name__}_create_state_parallel"
    backend = xb.get_backend("gpu")
    hlo_module = jaxpr_to_hlo_module(name, ClosedJaxpr(jaxpr, consts),
                                     donated_invars, backend)

    placement_specs, _ = tree_flatten(placement_specs)

    sharding_protos = []
    for spec in placement_specs:
        assert len(spec.mesh_ids) == 1
        sharding_protos.append(spec.sharding_specs[0].sharding_proto())

    xe.set_hlo_module_output_shardings(hlo_module, sharding_protos)
    physical_mesh = executable.physical_mesh

    run_spmd_partitioner_pass(hlo_module, physical_mesh.num_devices)
    strategy_config = executable.strategy_config

    return NormalMeshDriverExecutable(physical_mesh,
                                      hlo_module,
                                      strategy_config,
                                      avals,
                                      out_avals,
                                      [False] * len(avals))
