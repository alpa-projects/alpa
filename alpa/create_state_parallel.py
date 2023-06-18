"""Compile executables for creating training state distributedly."""
from collections import defaultdict, deque
from typing import Sequence, Optional

from jax.core import Var
from jax.interpreters import pxla
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef
import numpy as np

from alpa.device_mesh import ReplicatedDistributedArray, PhysicalDeviceMeshGroup, VirtualMeshGroup
from alpa.global_env import global_config
from alpa.mesh_executable import (NormalMeshDriverExecutable,
                                  GradAccMeshDriverExecutable)
from alpa.parallel_plan import PlacementSpec
from alpa.pipeline_parallel.compile_executable import compile_pipeshard_executable_internal
from alpa.pipeline_parallel.layer_construction import add_pipeline_marks_for_sliced_eqns
from alpa.pipeline_parallel.pipeshard_executable import PipeshardDriverExecutable
from alpa.pipeline_parallel.runtime_emitter import PipeshardConfig
from alpa.pipeline_parallel.stage_construction import UniformStageOption
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               AutoShardingOption)
from alpa.util import jaxpr_to_hlo, trace_jaxpr_with_micro_batch


class CreateStateExecutable(PipeshardDriverExecutable):
    """
    A distributed executable that creates a training state for a function
    parallelized by PipeshardParallel.
    """

    def __init__(self,
                 mesh_group: PhysicalDeviceMeshGroup,
                 virtual_mesh_group: VirtualMeshGroup,
                 pipeshard_config: PipeshardConfig,
                 target_placement_specs: Sequence[PlacementSpec],
                 in_tree: PyTreeDef,
                 out_tree: Optional[PyTreeDef] = None,
                 static_argnums: Optional[Sequence[int]] = None):
        super().__init__(mesh_group=mesh_group,
                         virtual_mesh_group= virtual_mesh_group,
                         pipeshard_config=pipeshard_config,
                         num_batch=1,
                         layer_option=None,
                         in_tree=in_tree,
                         out_tree=out_tree,
                         static_argnums=static_argnums)
        self.target_placement_specs = target_placement_specs

    def launch_on_driver(self, *args):
        outputs = super().launch_on_driver(*args)

        # Handle the creation of ReplicatedDistributedArray
        for idx, (array,
                  spec) in enumerate(zip(outputs, self.target_placement_specs)):
            assert array.device_mesh.mesh_id == spec.mesh_ids[0]
            assert array.indices == pxla.spec_to_indices(
                array.shape, spec.sharding_specs[0])

            if len(spec.mesh_ids) > 1:
                meshes = tuple(self.mesh_group[i] for i in spec.mesh_ids)
                distributed_arrays = [array]
                for mesh_id, sharding_spec in zip(spec.mesh_ids[1:],
                                                  spec.sharding_specs[1:]):
                    indices = pxla.spec_to_indices(array.shape, sharding_spec)
                    dis_array = self.mesh_group[mesh_id].shard_args_to_arrays(
                        (array.aval,), (indices,), (sharding_spec,),
                        (np.asarray(array),))[0]
                    distributed_arrays.append(dis_array)
                outputs[idx] = ReplicatedDistributedArray(
                    meshes, distributed_arrays)

        return outputs


def compile_create_state_executable(fun, in_tree, out_tree_thunk,
                                    static_argnums, donated_invars, train_step,
                                    other_args, *avals):
    # Trace to get jaxpr and HloModule
    closed_jaxpr, _ = trace_jaxpr_with_micro_batch(fun, [False] * len(avals), 1,
                                                   avals)
    out_avals = [v.aval for v in closed_jaxpr.jaxpr.outvars]
    jaxpr = closed_jaxpr.jaxpr

    name = f"{fun.__name__}_create_state_parallel"
    hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)

    # Compile train_step to get the placement specs.
    out_tree = out_tree_thunk()
    state_aval = tree_unflatten(out_tree, out_avals)
    executable = train_step.get_executable(state_aval, other_args)
    placement_specs = executable.get_input_placement_specs()[0]
    placement_specs, _ = tree_flatten(placement_specs)

    if (not isinstance(executable, NormalMeshDriverExecutable) and
            global_config.backend == "tpu"):
        raise NotImplementedError(f"{type(executable)} is not supported in tpu")
    if isinstance(executable,
                  (NormalMeshDriverExecutable, GradAccMeshDriverExecutable)):
        sharding_protos = []
        for spec in placement_specs:
            assert len(spec.mesh_ids) == 1
            sharding_protos.append(spec.sharding_specs[0].sharding_proto())

        physical_mesh = executable.physical_mesh

        # Run sharding propagation
        hlo.set_output_shardings(sharding_protos)
        hlo, stage_plan = run_auto_sharding_pass(
            hlo,
            physical_mesh.get_logical_mesh(
                executable.stage_plan.logical_mesh_shape), "single", 1,
            AutoShardingOption(enable_auto_sharding=False))

        return NormalMeshDriverExecutable(physical_mesh, hlo, stage_plan, avals,
                                          out_avals, [False] * len(avals),
                                          static_argnums, in_tree, out_tree)
    else:
        # Construct a new pipelined jaxpr
        outvars = jaxpr.outvars

        var2mesh = {}  # Dict[var -> mesh_id]
        eqn2mesh = {}  # Dict[eqn_idx -> mesh_id]

        output_shardings = []
        for var, spec in zip(outvars, placement_specs):
            if isinstance(var, Var):
                var2mesh[var] = spec.mesh_ids[0]
            output_shardings.append(spec.sharding_specs[0])

        num_meshes = len(executable.mesh_group)

        propagate_mesh_assignment(jaxpr, var2mesh, eqn2mesh)
        sliced_eqns = slice_jaxpr_with_mesh_assignment(jaxpr, eqn2mesh,
                                                       num_meshes)
        new_jaxpr = add_pipeline_marks_for_sliced_eqns(closed_jaxpr,
                                                       sliced_eqns)

        # Compile a pipeshard executable with predefined output shardings
        pipeshard_config, _ , virtual_mesh_group = compile_pipeshard_executable_internal(
            new_jaxpr, None, 1, [False] * len(avals), [False] * len(avals),
            executable.mesh_group.parent, 1, "inference",
            AutoShardingOption(enable_auto_sharding=False),
            UniformStageOption(), name, None, output_shardings, None, None)

        return CreateStateExecutable(mesh_group=executable.mesh_group,
                                     virtual_mesh_group= virtual_mesh_group,
                                     pipeshard_config=pipeshard_config,
                                     target_placement_specs=placement_specs,
                                     in_tree=in_tree,
                                     out_tree=out_tree_thunk(),
                                     static_argnums=static_argnums)


def propagate_mesh_assignment(jaxpr, var2mesh, eqn2mesh):
    """Propagate mesh assignment for all variables and equations.

    Note that this is different from the propagation in apply_grad.
    create_state_parallel: always assign one equation to one mesh.
      If one equation is used by multiple meshes, use send/recv to
      pass the value.
    apply_grad: can assign one equation to multiple meshes.
      If one equation is used by multiple meshes, replicate the
      computation on all meshes.
    """
    def_eqn = {}  # Dict[var -> eqn_idx]

    for idx, eqn in enumerate(jaxpr.eqns):
        for var in eqn.outvars:
            def_eqn[var] = idx

    mesh2vars = defaultdict(list)
    for var, mesh_idx in var2mesh.items():
        mesh2vars[mesh_idx].append(var)

    mesh_indices = list(mesh2vars.keys())
    mesh_indices.sort()

    for mesh_idx in mesh_indices:
        for var in mesh2vars[mesh_idx]:
            eqn_idx = def_eqn[var]
            if eqn_idx not in eqn2mesh:
                # Propagate from the definition equation to
                # all related equations
                queue = deque((eqn_idx,))

                while queue:
                    eqn_idx = queue.popleft()
                    eqn2mesh[eqn_idx] = mesh_idx

                    for var in jaxpr.eqns[eqn_idx].invars:
                        if isinstance(var, Var):
                            eqn_idx = def_eqn[var]
                            if eqn_idx not in eqn2mesh:
                                queue.append(eqn_idx)


def slice_jaxpr_with_mesh_assignment(jaxpr, eqn2mesh, num_meshes):
    sliced_eqns = [[] for _ in range(num_meshes)]

    for idx, eqn in enumerate(jaxpr.eqns):
        if idx in eqn2mesh:
            sliced_eqns[eqn2mesh[idx]].append(eqn)

    return sliced_eqns
