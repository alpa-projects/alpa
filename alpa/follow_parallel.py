"""Follow the parallelization strategy of another function."""
import logging

from jax._src.lib import xla_extension as xe
from jax.core import ClosedJaxpr
from jax.interpreters import partial_eval as pe
from jax.tree_util import tree_leaves

from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable)
from alpa.parallel_plan import PlacementSpec
from alpa.pipeline_parallel.compile_executable import (
    compile_pipeshard_executable)
from alpa.pipeline_parallel.layer_construction import (ManualLayerOption,
                                                       FollowLayerOption)
from alpa.pipeline_parallel.stage_construction import UniformStageOption
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               AutoShardingOption)
from alpa.util import (jaxpr_to_hlo_module, undefined_sharding_spec_proto)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def compile_follow_parallel_executable(fun, in_tree, out_tree_thunk,
                                       static_argnums, donated_invars,
                                       batch_invars, src_func,
                                       num_micro_batches, input_placement_specs,
                                       pipeline_schedule, layer_option, *avals):

    def is_leave(x):
        return isinstance(x, PlacementSpec) or x is None

    input_placement_specs = tree_leaves(input_placement_specs, is_leave)

    executable = src_func.get_last_executable()
    if isinstance(executable,
                  (NormalMeshDriverExecutable, GradAccMeshDriverExecutable)):
        if num_micro_batches != 1 and num_micro_batches is not None:
            logger.warning("num_micro_batches is ignored in FollowParallel")

        # Trace to get jaxpr and HloModule
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
        closed_jaxpr = ClosedJaxpr(jaxpr, consts)
        out_tree = out_tree_thunk()

        name = f"{fun.__name__}_follow_shard_parallel"
        hlo_module = jaxpr_to_hlo_module(name, closed_jaxpr, donated_invars)

        # Get input sharding specs
        sharding_protos = []
        for spec in input_placement_specs:
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
        elif layer_option == "follow":
            layer_option = FollowLayerOption(input_placement_specs,
                                             len(executable.mesh_group))
        else:
            raise ValueError(f"Invalid layer option: {layer_option}")

        input_shardings = [x.sharding_specs[0] for x in input_placement_specs]
        # TODO(lmzheng): handle ReplicatedDistributedArray, tied embedding
        mesh = executable.mesh_group.parent

        return compile_pipeshard_executable(
            fun, in_tree, out_tree_thunk, static_argnums, donated_invars,
            batch_invars, mesh, num_micro_batches, pipeline_schedule,
            AutoShardingOption(enable_auto_sharding=False), layer_option,
            UniformStageOption(), input_shardings, None, *avals)
