"""Follow the parallelization strategy of another function."""
import logging

from jax._src.lib import xla_bridge as xb, xla_extension as xe, xla_client as xc
from jax.core import ClosedJaxpr, Var
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_leaves

from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable,
                                  PlacementSpec)
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               AutoShardingOption)
from alpa.util import (jaxpr_to_hlo_module, trace_jaxpr_with_micro_batch,
                       undefined_sharding_spec_proto)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compile_follow_parallel_executable(fun, in_tree, out_tree_thunk,
                                       static_argnums, donated_invars,
                                       batch_invars, src_func,
                                       num_micro_batches, input_placement_specs,
                                       pipeline_schedule, layer_option, *avals):
    executable = src_func.get_last_executable()
    is_leave = lambda x: isinstance(x, PlacementSpec) or x is None
    placement_specs = tree_leaves(input_placement_specs, is_leave)

    if isinstance(executable,
                  (NormalMeshDriverExecutable, GradAccMeshDriverExecutable)):
        if num_micro_batches != 1 and num_micro_batches is not None:
            logger.warning("num_micro_batches is ignored in FollowParallel")

        # Trace to get jaxpr and HloModule
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        out_tree = out_tree_thunk()

        name = f"{fun.__name__}_follow_parallel"
        hlo_module = jaxpr_to_hlo_module(name, closed_jaxpr, donated_invars)

        # Get input sharding specs
        sharding_protos = []
        for spec in placement_specs:
            if spec is None:
                sharding_protos.append(undefined_sharding_spec_proto())
            else:
                assert len(spec.mesh_ids) == 1
                sharding_protos.append(spec.sharding_specs[0].sharding_proto())

        # Run sharding propagation
        physical_mesh = executable.physical_mesh
        xe.set_hlo_module_input_shardings(hlo_module, sharding_protos)
        hlo_module, stage_plan = run_auto_sharding_pass(
            hlo_module, physical_mesh.get_logical_mesh(), "single", 1,
            AutoShardingOption(enable_auto_sharding=False))

        return NormalMeshDriverExecutable(physical_mesh, hlo_module, stage_plan,
                                          avals, out_avals,
                                          [False] * len(avals), static_argnums,
                                          in_tree, out_tree)
    else:
        num_micro_batches = num_micro_batches or 1

        if layer_option == "manual":
            layer_option = ManualLayerOption()
        elif layer_option == "auto":
            layer_option = FollowLayerOption(placement_specs)
        else:
            raise ValueError(f"Invalid layer option: {layer_option}")

        return compile_pipeshard_executable(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, mesh, num_micro_batches, pipeline_schedule,
            AutoShardingOption(enable_auto_sharding=False),
            layer_option, UniformStageOption(),
            input_shardings, None, *avals)

#def slice_jaxpr_with_var_assignment(jaxpr, var2mesh):
#        # Trace to get jaxpr and HloModule
#        closed_jaxpr, batch_size = trace_jaxpr_with_micro_batch(
#            fun, batch_invars, num_micro_batches, avals)
#
#        if num_microbatch > 1:
#            # Trace again with a full batch
#            for store in fun.stores:
#                if store:
#                    store.reset()
#            full_batch_closed_jaxpr, _ = trace_jaxpr_with_micro_batch(
#                fun, batch_invars, 1, avals)
#        else:
#            full_batch_closed_jaxpr = None
#
#        # Construct a new pipelined jaxpr
#        invars = closed_jaxpr.jaxpr.invars
#
#        var2mesh = {}  # Dict[var -> mesh_id]
#
#        input_shardings = []
#        for var, spec in zip(invars, placement_specs):
#            if spec is None:
#                # Assign input vars to mesh 0 by default
#                if isinstance(var, Var):
#                    var2mesh[var] = 0
#            else:
#                if isinstance(var, Var):
#                    var2mesh[var] = spec.mesh_ids[0]
#                input_shardings.append(spec.sharding_specs[0])
#
#        num_meshes = len(executable.mesh_group)
#        sliced_eqns = slice_jaxpr_with_var_assignment(
#            closed_jaxpr.jaxpr, var2mesh)
#        new_jaxpr = add_pipeline_marks_for_sliced_eqns(
#            closed_jaxpr, sliced_eqns)
#
#
#
#    mesh_must_eqns = defaultdict(list)   # Dict[mesh_idx -> List[eqn_idx]]
#
#    # All direct users of an input var much be on the same mesh of the input var.
#    for idx, eqn in enumerate(jaxpr.eqns):
#        if eqn.primitive is pipeline_p:
#            raise ValueError("FollowParallel is not compatible with manual "
#                             "pipeline marker. Please do not insert manual "
#                             "pipeline marker in the function.")
#        for var in eqn.invars:
#            if var in var2mesh:
#                mesh_must_eqns[var2mesh[var]].append(idx)
#
#    cost_criteria = "flops"
#    costs = get_layer_construction_costs(jaxpr, cost_criteria=cost_criteria)
#    non_trivial, input_sizes, compute_costs = costs
#
#    max_cost = np.sum(compute_costs) * 10
#    for mesh_idx, eqn_indices in mesh_must_eqns:
#        tmp_cost = max_cost / len(eqn_indices)
#        for eqn_idx in eqn_indices:
#            compute_costs[eqn_idx] = tmp_cost
#
#    sliced_eqns, _ = cluster_jaxpr_by_cost(jaxpr,
#					   layer_num=len(mesh_must_eqns),
#					   eps=0.1,
#					   costs,
#					   cost_criteria=cost_criteria)
#    return sliced_eqns
#
