"""Compile executables for creating training state distributedly."""
from collections import defaultdict, deque

from jax._src.lib import xla_bridge as xb, xla_extension as xe
from jax.core import (Jaxpr, ClosedJaxpr, Literal, new_jaxpr_eqn, gensym,
                      get_aval, raise_to_shaped, AbstractValue, Var)
from jax.interpreters import partial_eval as pe, pxla
from jax.tree_util import tree_flatten, tree_unflatten, PyTreeDef

from alpa.device_mesh import ReplicatedDistributedArray
from alpa.mesh_executable import NormalMeshDriverExecutable, GradAccMeshDriverExecutable
from alpa.measure_record import StrategyConfig
from alpa.pipeline_parallel.compile_executable import compile_pipeshard_executable_internal
from alpa.pipeline_parallel.layer_construction import add_pipeline_marks_for_sliced_eqns
from alpa.pipeline_parallel.stage_construction import UniformStageOption
from alpa.shard_parallel.auto_sharding import run_spmd_partitioner_pass, AutoShardingOption
from alpa.util import jaxpr_to_hlo_module, OrderedSet


class CreateStateExecutable:
    def __init__(self, executable, placement_specs):
        self.executable = executable
        self.mesh_group = executable.mesh_group
        self.placement_specs = placement_specs

    def launch_on_driver(self, *args):
        outputs = self.executable.launch_on_driver(*args)

        for idx, spec in enumerate(self.placement_specs):
            if len(spec.mesh_ids) > 1:
                src = outputs[idx]
                meshes = tuple(self.mesh_group[i] for i in spec.mesh_ids)
                distributed_arrays = [src]
                for mesh_id, sharding_spec in zip(spec.mesh_ids[1:], spec.sharding_specs[1:]):
                    indices = pxla.spec_to_indices(src.shape, sharding_spec)
                    dis_array = self.mesh_group[mesh_id].shard_args_to_arrays(
                        (src.aval,), (indices,), (sharding_spec,), (src,))[0]
                    distributed_arrays.append(dis_array)
                outputs[idx] = ReplicatedDistributedArray(meshes, distributed_arrays)

        return outputs


def compile_create_state_executable(fun, in_tree, out_tree_thunk, static_argnums,
                                    donated_invars, batch_invars, train_step,
                                    other_args, *avals):
    # Trace to get jaxpr
    jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    out_tree = out_tree_thunk()
    state_aval = tree_unflatten(out_tree, out_avals)

    # Compile train_step to get the placement specs.
    executable = train_step.get_executable(state_aval, other_args)
    placement_specs = executable.get_placement_specs()[0]

    name = f"{fun.__name__}_create_state_parallel"
    backend = xb.get_backend("gpu")
    hlo_module = jaxpr_to_hlo_module(name, closed_jaxpr,
                                     donated_invars, backend)
    placement_specs, _ = tree_flatten(placement_specs)

    if isinstance(executable, (NormalMeshDriverExecutable, GradAccMeshDriverExecutable)):
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
    else:
        # Construct a new pipelined jaxpr
        outvars = jaxpr.outvars

        var2mesh = {}  # Dict[var -> mesh_id]
        eqn2mesh = {}  # Dict[eqn_idx -> mesh_id]

        for var, spec in zip(outvars, placement_specs):
            if isinstance(var, Var):
                var2mesh[var] = spec.mesh_ids[0]

        num_meshes = len(executable.mesh_group)

        propagate_mesh_assignment(jaxpr, var2mesh, eqn2mesh)
        eqns = slice_jaxpr_with_mesh_assignment(jaxpr, eqn2mesh, num_meshes)
        new_jaxpr = add_pipeline_marks_for_sliced_eqns(closed_jaxpr, eqns)

        # Compile a pipeshard executable
        executable = compile_pipeshard_executable_internal(
            new_jaxpr, None, 1, in_tree, out_tree_thunk,
            None, [False] * len(avals), [False] * len(avals),
            executable.mesh_group.parent, 1,
            "inference", AutoShardingOption(),
            UniformStageOption())
        return CreateStateExecutable(executable, placement_specs)


def propagate_mesh_assignment(jaxpr, var2mesh, eqn2mesh):
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
    eqns = [[] for _ in range(num_meshes)]

    for idx, eqn in enumerate(jaxpr.eqns):
        if idx in eqn2mesh:
            eqns[eqn2mesh[idx]].append(eqn)

    return eqns
