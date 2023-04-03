"""Pipeline computation definitions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Sequence, Any, Dict, Optional

from jax import jit
from jax.lib import (
    xla_bridge as xb,
    xla_extension as xe,
)
from jax._src.util import partial, safe_map
from jax._src import dispatch
from jax.core import (Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal,
                      jaxpr_as_fun, gensym, named_call_p, ShapedArray)
from jax.interpreters import pxla
import numpy as np

from alpa.mesh_executable import PartialGradAccMeshDriverExecutable
from alpa.parallel_plan import StagePlan
from alpa.pipeline_parallel.primitive_def import (mark_hook_jaxpreqn,
                                                  pipeline_p,
                                                  mark_pipeline_jaxpreqn)
from alpa.shard_parallel.auto_sharding import (run_auto_sharding_pass,
                                               run_spmd_partitioner_pass,
                                               get_input_output_sharding_specs,
                                               hlo_sharding_to_sharding_spec,
                                               AutoShardingOption)
from alpa.global_env import global_config
from alpa.util import (OrderedSet, clone_jaxpr, clone_jaxpr_eqn,
                       get_compile_options, jaxpr_to_hlo,
                       setup_computation_alias, compile_dummy_zero_constant,
                       get_var_mapping, undefined_sharding_spec_proto,
                       new_jaxpr_eqn, replicated_sharding_spec_proto)
from alpa.wrapped_hlo import HloStatus, WrappedHlo

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PipelineComputation(ABC):
    """
    Base class of pipeline computations.

    Attributes:
        name (str): The name of the pipeline computation.
        invars (Sequence[Var]): The list of input variables, corresponding to
            the order of the runnable inputs.
        outvars (Sequence[Var]): The list of output variables, corresponding to
            the order of the runnable outputs.
    """

    name: str
    invars: Sequence[Var] = field(default_factory=list)
    outvars: Sequence[Var] = field(default_factory=list)

    @abstractmethod
    def get_runnable(self, mesh=None):
        """Compile the computation and get the runnable."""
        raise NotImplementedError()


@dataclass
class StrVarPipelineComputation:
    """Stringified computation with all Set/Dict have string keys."""

    name: str
    invars: Sequence[str]
    outvars: Sequence[str]

    @classmethod
    def from_pipeline_computation(cls,
                                  pipeline_computation: PipelineComputation):
        """Construct a StrVarPipelineComputation from a PipelineComputation."""
        return cls(
            name=pipeline_computation.name,
            invars=[repr(var) for var in pipeline_computation.invars],
            outvars=[repr(var) for var in pipeline_computation.outvars],
        )


@dataclass
class JaxPipelineComputation(PipelineComputation):
    """
    A pipeline computation defined by Jaxpr.

    Attributes:
        eqns (Sequence[JaxprEqn]): Jaxpr equations of the pipeline computation.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            computation.
    """

    eqns: Sequence[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)

    def closed_jaxpr(self) -> ClosedJaxpr:
        """
        Get the closed Jaxpr of the pipeline computation.

        Returns:
            ClosedJaxpr: The result ClosedJaxpr.
        """
        jaxpr = Jaxpr(
            constvars=list(self.consts_dir.keys()),
            invars=self.invars,
            outvars=self.outvars,
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, list(self.consts_dir.values()))
        return closed_jaxpr

    def get_runnable(self, mesh=None):
        """Return a JIT callable of the pipeline computation."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))

    @classmethod
    def from_closed_jaxpr(cls, name, closed_jaxpr: ClosedJaxpr):
        """Construct a JaxPipelineComputation from a Jaxpr."""
        return cls(name=name,
                   invars=closed_jaxpr.jaxpr.invars,
                   outvars=closed_jaxpr.jaxpr.outvars,
                   eqns=closed_jaxpr.eqns,
                   consts_dir=dict(
                       zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts)))

    def outvars_def_order(self):
        """
        Get the order of outvars by when they are defined in the jaxpr. This may
        be not accurate because XLA optimizations may reorder it, but we only
        focus on the order of activations which have data dependency so it's ok.
        """
        outvars = self.outvars
        assert self.eqns[-1].primitive is pipeline_p
        assert tuple(self.eqns[-1].outvars) == tuple(outvars)
        pre_marker_vars = self.eqns[-1].invars
        pre_marker_vars = {v: idx for idx, v in enumerate(pre_marker_vars)}
        final_order = []
        for inv in self.invars:
            if inv in pre_marker_vars:
                final_order.append(pre_marker_vars[inv])
        for eqn in self.eqns:
            for var in eqn.outvars:
                if not isinstance(var, DropVar) and var in pre_marker_vars:
                    final_order.append(pre_marker_vars[var])
        assert len(final_order) == len(outvars)
        return [outvars[idx] for idx in final_order]


@dataclass
class XlaPipelineComputation(PipelineComputation):
    """A pipeline computation defined by XLA HLO Module."""

    hlo: WrappedHlo = None

    @classmethod
    def from_jax_pipeline_computation(
            cls, jax_pipeline_computation: JaxPipelineComputation):
        """
        Construct a XlaPipelineComputation from a JaxPipelineComputation.

        Args:
            jax_pipeline_computation (JaxPipelineComputation): the source
              JaxPipelineComputation.
        """
        closed_jaxpr = jax_pipeline_computation.closed_jaxpr()
        name = f"pipeline_computation_{jax_pipeline_computation.name}"
        donated_invars = (False,) * len(jax_pipeline_computation.invars)
        hlo = jaxpr_to_hlo(name, closed_jaxpr, donated_invars)

        return cls(
            name=jax_pipeline_computation.name,
            hlo=hlo,
            invars=jax_pipeline_computation.invars,
            outvars=jax_pipeline_computation.outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""
        out_avals = [var.aval for var in self.outvars]
        tuple_args = len(self.invars) > 100 and global_config.backend == "tpu"
        backend = xb.get_backend(global_config.backend)
        device = backend.get_default_device_assignment(1)[0]
        options = get_compile_options(
            num_replicas=1,
            num_partitions=1,
            device_assignment=(device.id,) if device else None,
            use_spmd_partitioning=False,
            parameter_is_tupled_arguments=tuple_args,
            build_random_seed=global_config.compile_random_seed,
        )

        xla_computation = self.hlo.get_computation()
        compiled = backend.compile(xla_computation, compile_options=options)
        self.hlo.module = compiled.hlo_modules()[0]
        self.hlo.status = HloStatus.FULLY_OPTIMIZED
        # pylint: disable=protected-access
        result_handler = dispatch._result_handler(backend, device, [(
            aval,
            True,
        ) for aval in out_avals])
        buffer_counts = (None if len(out_avals) == 1 else [
            dispatch.aval_to_num_buffers(aval) for aval in out_avals
        ])
        kept_var_idx = range(len(self.invars))
        return partial(
            dispatch._execute_compiled,  # pylint: disable=protected-access
            self.name,
            compiled,
            None,
            buffer_counts,
            result_handler,
            False,
            (),
            kept_var_idx,
            False)

    def get_hlo_text(self):
        """Get the HLO text."""
        return self.hlo.to_string()


@dataclass
class XlaShardedPipelineComputation(PipelineComputation):
    """
    A pipeline computation defined by XLA HLO Module.
    The XLA HLO is annotated by sharding spec.
    """

    hlo: WrappedHlo = None
    donated_invars: Sequence[bool] = None
    stage_plan: StagePlan = None
    input_sharding_specs: Sequence[pxla.ShardingSpec] = None
    output_sharding_specs: Sequence[pxla.ShardingSpec] = None
    output_acc_grad_indices: Sequence[int] = None
    donatables: OrderedSet[Var] = None

    @classmethod
    def dummy_computation(cls, name, logical_mesh_shape, gensym_func):
        """Create a dummy computation."""
        stage_plan = StagePlan(global_config.compile_random_seed,
                               logical_mesh_shape, 1, 1, AutoShardingOption(),
                               None, 0)
        sharding_annotated_hlo = compile_dummy_zero_constant()
        outvar = gensym_func(ShapedArray((), np.dtype(np.int32)))
        return cls(
            name=name,
            hlo=sharding_annotated_hlo,
            stage_plan=stage_plan,
            donated_invars=[],
            invars=[],
            outvars=[outvar],
            output_acc_grad_indices=[],
            donatables=OrderedSet(),
        )

    @classmethod
    def from_auto_sharded_computation(
            cls,
            *,
            jax_pipeline_computation: JaxPipelineComputation,
            sharding_annotated_hlo: WrappedHlo,
            stage_plan: StagePlan,
            donated_invars: Sequence[bool] = None,
            acc_grad_outvars: Sequence[Var] = (),
            donatables: OrderedSet[Var] = None):
        """Run auto-sharding optimizer on a Jax pipeline computation."""
        if donatables is None:
            donatables = OrderedSet()

        if not donated_invars:
            donated_invars = (False,) * len(jax_pipeline_computation.invars)

        acc_grad_indices = [
            out_idx
            for out_idx, outvar in enumerate(jax_pipeline_computation.outvars)
            if outvar in acc_grad_outvars
        ]

        return cls(name=jax_pipeline_computation.name,
                   hlo=sharding_annotated_hlo,
                   stage_plan=stage_plan,
                   donated_invars=donated_invars,
                   invars=jax_pipeline_computation.invars,
                   outvars=jax_pipeline_computation.outvars,
                   output_acc_grad_indices=acc_grad_indices,
                   donatables=donatables)

    def donate_intermediates(self, computation):
        """Donate intermediate variables."""
        # FIXME (yonghao): this function is not being used.
        # get sharding annotated hlo module
        hlo_module = computation.as_hlo_module()
        donatable = OrderedSet(self.donatables)
        # get sharding specs
        hlo_module.infer_spmd_shardings()
        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        logical_mesh_shape = self.stage_plan.logical_mesh_shape
        input_shardings = hlo_module.spmd_parameters_shardings()
        input_sharding_specs = [
            hlo_sharding_to_sharding_spec(proto_tuple, aval, logical_mesh_shape)
            for (proto_tuple, aval) in zip(input_shardings, avals)
        ]
        output_shardings = hlo_module.spmd_output_sharding()
        output_sharding_specs = hlo_sharding_to_sharding_spec(
            output_shardings, out_avals, logical_mesh_shape)

        num_donated = np.count_nonzero(self.donated_invars)
        donatable_outvars = OrderedSet(self.outvars[num_donated:])
        donated_invars = []
        donated_outvars = []
        var_indices = dict(zip(self.outvars, range(len(self.outvars))))
        var_indices.update(dict(zip(self.invars, range(len(self.invars)))))
        for idx, invar in enumerate(self.invars):
            if invar not in donatable:
                # not donatable
                continue
            if self.donated_invars[idx]:
                # already donated
                continue
            for outvar in donatable_outvars:
                if (invar.aval.shape == outvar.aval.shape and
                        input_sharding_specs[var_indices[invar]]
                        == output_sharding_specs[var_indices[outvar]]):
                    donated_invars.append(invar)
                    donated_outvars.append(outvar)
                    donatable_outvars.discard(outvar)
                    break
        # set alias
        for invar, outvar in zip(donated_invars, donated_outvars):
            invar_idx, outvar_idx = var_indices[invar], var_indices[outvar]
            computation.setup_alias((outvar_idx,), invar_idx, ())
        for invar in donated_invars:
            self.donated_invars[var_indices[invar]] = True

    def get_spmd_partitioned(self):
        """Run spmd partitioner to get the input/output sharding specs after
        partitioning."""
        if self.hlo.is_spmd_partitioned():
            return self.hlo

        stage_plan = self.stage_plan
        logical_mesh_shape = stage_plan.logical_mesh_shape
        setup_computation_alias(self.hlo, self.donated_invars)

        num_devices = np.prod(logical_mesh_shape)
        rewrite_for_grad_acc = len(self.output_acc_grad_indices) > 0
        hlo = run_spmd_partitioner_pass(
            self.hlo,
            num_devices,
            rewrite_for_grad_acc=rewrite_for_grad_acc,
            rewrite_grad_acc_indices=self.output_acc_grad_indices)

        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        input_sharding_specs, output_sharding_specs = (
            get_input_output_sharding_specs(hlo.get_module(), avals, out_avals,
                                            num_devices,
                                            stage_plan.logical_mesh_shape))
        self.input_sharding_specs = input_sharding_specs
        self.output_sharding_specs = output_sharding_specs
        # The run_spmd_partitioner_pass modifies hlo module in-place,
        # so the old hlo module cannot be accessed anymore
        return hlo

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""
        if not mesh:
            raise RuntimeError(
                "`XlaShardedPipelineComputation` requires a mesh.")
        hlo = self.get_spmd_partitioned()

        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        mesh_executable = PartialGradAccMeshDriverExecutable(
            mesh, hlo, self.stage_plan, avals, out_avals, self.donated_invars)
        return mesh_executable.get_driver_callable()

    def get_hlo_text(self):
        """Get the HLO text."""
        assert self.hlo.is_sharding_annotated()
        return self.hlo.to_string()


def slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr: ClosedJaxpr) -> Sequence[JaxPipelineComputation]:
    """Slice a closed jaxpr into multiple JaxPipelineComputation by full
    pipeline markers."""
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_computations = []
    current_computation = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params["mark_type"] == "start":
            assert current_computation is None, (
                "Defining a pipeline computation "
                "inside a pipeline computation is "
                "not allowed.")
            current_computation = JaxPipelineComputation(
                name=eqn.params["name"])
            for var in eqn.invars:
                if isinstance(var, Literal):
                    pass
                elif var in global_consts_dir:
                    current_computation.consts_dir[var] = global_consts_dir[var]
                else:
                    current_computation.invars.append(var)

        for var in eqn.invars:
            if not isinstance(var, Literal) and var in global_consts_dir:
                current_computation.consts_dir[var] = global_consts_dir[var]

        assert current_computation is not None
        current_computation.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params["mark_type"] == "end":
            assert current_computation is not None, (
                "Ending a pipeline computation before its start.")
            assert current_computation.name == eqn.params["name"], (
                "Ending a pipeline computation different from its start.")
            for var in eqn.outvars:
                current_computation.outvars.append(var)
            result_computations.append(current_computation)
            current_computation = None

    return result_computations


def mark_missing_vars_in_backward_computation_pipeline_marks(
        computations: Sequence[JaxPipelineComputation], global_invars,
        global_outvars, gensym_func):
    """
    Fix missing vars generated by jax.grad and alpa.grad.

    Fix missing input variables in pipeline markers of stages generated by
    jax.grad or alpa.grad. Also remove unused variables in the pipeline
    markers.
    """
    assert len(computations) % 2 == 0.
    num_forward_computations = len(computations) // 2

    var_computation_id = {}
    for var in global_invars:
        if not isinstance(var, Literal):
            var_computation_id[var] = -1

    computation_marked_to_unmarked_invars = [{} for _ in computations]
    computation_weight_invars = [{} for _ in computations]
    computation_additional_invars = [OrderedSet() for _ in computations]
    computation_additional_outvars = [OrderedSet() for _ in computations]
    for computation_id, computation in enumerate(computations):
        for eqn in computation.eqns:
            if eqn.primitive == pipeline_p and eqn.params[
                    "mark_type"] == "start":
                for invar, outvar in zip(eqn.invars, eqn.outvars):
                    computation_marked_to_unmarked_invars[computation_id][
                        outvar] = invar
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in computation.consts_dir and
                        var not in computation.invars):
                    source_computation_id = var_computation_id[var]
                    if source_computation_id != computation_id:
                        # Special case for the model weights. If a backward
                        # computation is using an invar of a forward
                        # computation, do not let the invar go into the stage.
                        # Instead, we can directly use the original invar.
                        if (computation_id >= num_forward_computations and
                                source_computation_id
                                == 2 * num_forward_computations -
                                computation_id - 1 and
                                var in computation_marked_to_unmarked_invars[
                                    source_computation_id]):
                            computation_weight_invars[computation_id][var] = (
                                computation_marked_to_unmarked_invars[
                                    source_computation_id][var])
                            continue
                        # Mark all the variables in the backward computation
                        # that are not currently defined in pipeline markers.
                        if (source_computation_id != -1 and var not in
                                computations[source_computation_id].outvars):
                            computation_additional_outvars[
                                source_computation_id].add(var)
                        computation_additional_invars[computation_id].add(var)
            for var in eqn.outvars:
                var_computation_id[var] = computation_id

    for var in global_outvars:
        source_computation_id = var_computation_id[var]
        if source_computation_id != -1 and var not in computations[
                source_computation_id].outvars:
            computation_additional_outvars[source_computation_id].add(var)

    new_computations = []

    for i, computation in enumerate(computations):
        assert (computation.eqns[0].primitive is pipeline_p and
                computation.eqns[0].params["mark_type"] == "start")
        assert (computation.eqns[-1].primitive is pipeline_p and
                computation.eqns[-1].params["mark_type"] == "end")
        new_computation = JaxPipelineComputation(
            computation.name, consts_dir=computation.consts_dir)

        computation_var_mapping = {
            var: gensym_func(var.aval)
            for var in computation_additional_invars[i] |
            computation_additional_outvars[i] |
            computation_weight_invars[i].keys()
        }
        pipeline_start_invars = list(computation.eqns[0].invars)
        pipeline_start_outvars = [
            get_var_mapping(computation_var_mapping, var)
            for var in computation.eqns[0].outvars
        ]
        new_computation.invars = list(computation.invars)
        for var in computation_additional_invars[i]:
            pipeline_start_invars.append(var)
            pipeline_start_outvars.append(computation_var_mapping[var])
        for marked_var, unmarked_var in computation_weight_invars[i].items():
            pipeline_start_invars.append(unmarked_var)
            pipeline_start_outvars.append(computation_var_mapping[marked_var])
        pipeline_start_invars_without_literal = []
        pipeline_start_outvars_without_literal = []
        for invar, outvar in zip(pipeline_start_invars, pipeline_start_outvars):
            if isinstance(invar, Literal):
                computation_var_mapping[outvar] = invar
            else:
                pipeline_start_invars_without_literal.append(invar)
                pipeline_start_outvars_without_literal.append(outvar)
        new_computation.invars = list(pipeline_start_invars_without_literal)
        new_computation.eqns.append(computation.eqns[0]._replace(
            invars=pipeline_start_invars_without_literal,
            outvars=pipeline_start_outvars_without_literal))

        for eqn in computation.eqns[1:-1]:
            invars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.invars
            ]
            outvars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.outvars
            ]
            new_computation.eqns.append(
                eqn._replace(invars=invars, outvars=outvars))

        pipeline_end_invars = [
            get_var_mapping(computation_var_mapping, var)
            for var in computation.eqns[-1].invars
        ]
        pipeline_end_outvars = list(computation.eqns[-1].outvars)
        for var in computation_additional_outvars[i]:
            pipeline_end_invars.append(computation_var_mapping[var])
            pipeline_end_outvars.append(var)
        pipeline_end_invars_without_dropvar = []
        pipeline_end_outvars_without_dropvar = []
        for invar, outvar in zip(pipeline_end_invars, pipeline_end_outvars):
            if not isinstance(outvar, DropVar):
                pipeline_end_invars_without_dropvar.append(invar)
                pipeline_end_outvars_without_dropvar.append(outvar)
        new_computation.outvars = list(pipeline_end_outvars_without_dropvar)
        new_computation.eqns.append(computation.eqns[-1]._replace(
            invars=pipeline_end_invars_without_dropvar,
            outvars=pipeline_end_outvars_without_dropvar))
        new_computations.append(new_computation)

    return new_computations


def pipeline_dce(jax_pipeline_computations: Sequence[JaxPipelineComputation],
                 global_outvars):
    """
    Clear unused vars cross pipeline computations.

    This function removes grad and only keeps accumulated grad.
    """

    def dce_pipe_marker(marker: JaxprEqn, used_set):
        kept_indices = [
            i for i, var in enumerate(marker.outvars) if var in used_set
        ]
        new_marker = mark_pipeline_jaxpreqn(
            [marker.invars[i] for i in kept_indices],
            [marker.outvars[i] for i in kept_indices], marker.params["name"],
            marker.params["mark_type"])
        return new_marker

    global_used = OrderedSet(global_outvars)
    new_computations = []
    for computation in reversed(jax_pipeline_computations):
        new_eqns = []
        # handle pipe end
        pipe_end = computation.eqns[-1]
        assert (pipe_end.primitive is pipeline_p and
                pipe_end.params["mark_type"]
                == "end"), "computation not ended by a pipeline marker"
        new_pipe_end = dce_pipe_marker(pipe_end, global_used)
        new_eqns.append(new_pipe_end)
        # handle normal instructions
        local_used = OrderedSet(new_pipe_end.invars)
        for eqn in reversed(computation.eqns[1:-1]):
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar) and outvar in local_used:
                    new_eqns.append(eqn)
                    local_used.update([
                        invar for invar in eqn.invars if isinstance(invar, Var)
                    ])
                    break
        # handle pipe start
        pipe_start = computation.eqns[0]
        assert (pipe_start.primitive is pipeline_p and
                pipe_start.params["mark_type"]
                == "start"), "computation not started by a pipeline marker"
        new_pipe_start = dce_pipe_marker(pipe_start, local_used)
        new_eqns.append(new_pipe_start)
        global_used.update(new_pipe_start.invars)

        new_eqns = list(reversed(new_eqns))
        new_computation = JaxPipelineComputation(
            computation.name,
            invars=new_pipe_start.invars,
            outvars=new_pipe_end.outvars,
            eqns=new_eqns,
            consts_dir=computation.consts_dir)
        new_computations.append(new_computation)
    new_computations = list(reversed(new_computations))
    return new_computations


def rearrange_vars(invars,
                   selected: Sequence[Var],
                   pipe_marker=None,
                   is_input=True):
    """
    Rearrange vars to let those in selected be first.

    If the pipe_marker is given, rearrange invars and outvars in pipemarker as
    well.

    Args:
        invars (Sequence[Var]): all vars to be rearranged.
        selected (Sequence[Var]): vars selected to be prior.
        pipe_marker (JaxprEqn): pipe marker corresponding to vars
        is_input (bool): the var is input of pipe_marker, if False, it is output
    """
    new_vars = list(selected)
    selected = OrderedSet(selected)
    for var in invars:
        if var not in selected:
            new_vars.append(var)

    if pipe_marker is None:
        return new_vars

    if is_input:
        new_invars = list(new_vars)
        var_set = set(new_vars)
        # the pipeline start marker also include constvars
        for v in pipe_marker.invars:
            if v not in var_set:
                new_invars.append(v)
        invar_idx = {v: idx for idx, v in enumerate(pipe_marker.invars)}
        new_outvars = [
            pipe_marker.outvars[invar_idx[var]] for var in new_invars
        ]
    else:
        new_outvars = list(new_vars)
        outvar_idx = {v: idx for idx, v in enumerate(pipe_marker.outvars)}
        new_invars = [
            pipe_marker.invars[outvar_idx[var]] for var in new_outvars
        ]
    new_marker = clone_jaxpr_eqn(pipe_marker, new_invars, new_outvars)
    return new_vars, new_marker


def generate_computations_from_modules(
        jax_computations, computation_names, computation_hlos, donate_invars,
        donatable_lists, acc_grad_outvars,
        stage_plan) -> Sequence[XlaShardedPipelineComputation]:
    """Generate pipeline computation from HLO modules."""
    module_dict = dict(zip(computation_names, computation_hlos))
    computations = [
        XlaShardedPipelineComputation.from_auto_sharded_computation(
            sharding_annotated_hlo=module_dict[computation.name],
            jax_pipeline_computation=computation,
            stage_plan=stage_plan,
            donated_invars=donate_invars,
            acc_grad_outvars=acc_grad_outvars,
            donatables=donatables)
        for computation, donate_invars, donatables in zip(
            jax_computations, donate_invars, donatable_lists)
    ]
    return computations


def generate_sharded_xla_computations_arguments(
        name: str, jax_computations: Sequence[JaxPipelineComputation],
        computation_donate_invars: Sequence[bool],
        input_sharding_dict: Dict[Var, pxla.ShardingSpec],
        output_sharding_dict: Dict[Var, pxla.ShardingSpec],
        stage_input_sharding: Optional[Sequence[pxla.ShardingSpec]]):
    """
    Generates the arguments for distributed compilation.

    Similar to generate_sharded_xla_computations but only generate arguments.
    """
    invars = OrderedSet()
    outvars = OrderedSet()
    donation_mapping = {}
    eqns = []
    consts_dir = {}
    for computation, donation in zip(jax_computations,
                                     computation_donate_invars):
        consts_dir.update(computation.consts_dir)
        # Do not add local invars into the invars
        invars.update([var for var in computation.invars if var not in outvars])
        outvars.update(computation.outvars)
        for idx, var in enumerate(computation.invars):
            if not donation[idx] or var not in invars:
                continue
            donation_mapping[computation.invars[idx]] = computation.outvars[idx]
        eqns += computation.eqns
    invars = rearrange_vars(invars, donation_mapping.keys())
    outvars = rearrange_vars(outvars, donation_mapping.values())
    jaxpr = Jaxpr(
        constvars=list(consts_dir.keys()),
        invars=list(invars),
        outvars=list(outvars),
        eqns=eqns,
    )

    donation_num = len(donation_mapping)
    dummy_donated_invars = (True,) * donation_num + (False,) * (len(invars) -
                                                                donation_num)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts_dir.values())
    hlo = jaxpr_to_hlo(name, closed_jaxpr, dummy_donated_invars)

    if input_sharding_dict:
        sharding_protos = []
        for x in invars:
            spec = input_sharding_dict.get(x, None)
            if spec is None:
                sharding_protos.append(undefined_sharding_spec_proto())
            else:
                sharding_protos.append(spec.sharding_proto())
        hlo.set_input_shardings(sharding_protos)

    if output_sharding_dict:
        sharding_protos = []
        for x in outvars:
            spec = output_sharding_dict.get(x, None)
            if spec is None:
                sharding_protos.append(replicated_sharding_spec_proto())
            else:
                sharding_protos.append(spec.sharding_proto())
        hlo.set_output_shardings(sharding_protos)

    if stage_input_sharding:
        sharding_protos = [
            sharding_spec.sharding_proto()
            for sharding_spec in stage_input_sharding
        ]
        hlo.set_input_shardings(sharding_protos)

    flops = xe.hlo_module_count_flop_dot_conv_only(hlo.get_module())
    return hlo, flops


def generate_sharded_xla_computations(
        name: str, jax_computations: Sequence[JaxPipelineComputation],
        computation_donate_invars, donatable_lists, acc_grad_outvars,
        num_micro_batches, logical_mesh, autosharding_option,
        input_sharding_dict, output_sharding_dict, stage_input_sharding):
    """
    Generate sharded XLA computations.

    It runs the auto-sharding pass on the given JaxPipelineComputations.
    Note: we merge the co-located forward and backward computation and compile
    them together to get a sharding strategy config.
    """
    hlo, flops = generate_sharded_xla_computations_arguments(
        name, jax_computations, computation_donate_invars, input_sharding_dict,
        output_sharding_dict, stage_input_sharding)

    #  pylint: disable=unbalanced-tuple-unpacking
    (computation_names, computation_hlos,
     stage_plan) = run_auto_sharding_pass(hlo, logical_mesh, "stages",
                                          num_micro_batches,
                                          autosharding_option)

    computations = generate_computations_from_modules(
        jax_computations, computation_names, computation_hlos,
        computation_donate_invars, donatable_lists, acc_grad_outvars,
        stage_plan)
    return computations, flops


def rewrite_hook(eqns, gensym_fn):
    """ (Deprecated because we now profile forward and backward separately)
    Rewrite the hook marker to include the intermediate variables.

    Assume there is a special "hook" marker eqn in eqns that devide the
    eqns into two parts. This function rewrites the hook to capture all the
    variables that are passed between the two parts.
    """
    for idx, eqn in enumerate(eqns):
        eqn: JaxprEqn
        if ("mark_type" in eqn.params and eqn.params["mark_type"] == "hook"):
            used_vars = OrderedSet()
            defined_vars = OrderedSet()
            for e in eqns[0:idx]:
                defined_vars.update(
                    [v for v in e.outvars if not isinstance(v, DropVar)])
            for e in eqns[idx + 1:]:
                used_vars.update([v for v in e.invars if isinstance(v, Var)])
            marked = used_vars.intersection(defined_vars)
            hooked = list(marked)
            new_hook = mark_hook_jaxpreqn(hooked,
                                          [gensym_fn(v.aval) for v in hooked])
            rewrite_dict = dict(zip(hooked, new_hook.outvars))
            eqns[idx] = new_hook
            for i in range(idx + 1, len(eqns)):
                e = eqns[i]
                eqns[i] = clone_jaxpr_eqn(
                    e, [get_var_mapping(rewrite_dict, v) for v in e.invars])
            return new_hook
    return None


def _wrap_with_call(closed_jaxpr: ClosedJaxpr, invars, outvars, name):
    new_invars = closed_jaxpr.jaxpr.invars + closed_jaxpr.jaxpr.constvars
    jaxpr = clone_jaxpr(closed_jaxpr, new_invars, constvars=[], consts=[]).jaxpr
    params = dict(name=name, call_jaxpr=jaxpr)
    return new_jaxpr_eqn(invars + closed_jaxpr.jaxpr.constvars, outvars,
                         named_call_p, params)


def _rearrange_in_out_for_donation(invars, outvars, donation_map):
    outvar_set = set(outvars)
    donated_invars = [
        var for var in invars
        if (var in donation_map and donation_map[var] in outvar_set)
    ]
    donated_outvars = [donation_map[var] for var in donated_invars]
    invars = rearrange_vars(invars, donated_invars)
    outvars = rearrange_vars(outvars, donated_outvars)
    num_donated = len(donated_invars)
    return invars, outvars, num_donated


def merge_unmarked_with_call(jaxprs: Sequence[ClosedJaxpr],
                             names: Sequence[str],
                             outvars,
                             donation_map=None):
    """Merge a sequence of jaxprs (no pipeline marker) using named call."""
    gensym_fn = gensym([closed_jaxpr.jaxpr for closed_jaxpr in jaxprs])
    eqns = []
    invars = OrderedSet()
    intermediates = OrderedSet()
    const_dir = {}
    for stage_name, closed_jaxpr in zip(names, jaxprs):
        invars.update(closed_jaxpr.jaxpr.invars)
        intermediates.update(closed_jaxpr.jaxpr.outvars)
        const_dir.update(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
        jaxpr = closed_jaxpr.jaxpr

        sym_invars = [gensym_fn(var.aval) for var in jaxpr.invars]
        sym_outvars = [gensym_fn(var.aval) for var in jaxpr.outvars]
        eqns.append(
            mark_pipeline_jaxpreqn(jaxpr.invars, sym_invars, stage_name,
                                   "start"))
        eqns.append(
            _wrap_with_call(closed_jaxpr, sym_invars, sym_outvars, stage_name))
        eqns.append(
            mark_pipeline_jaxpreqn(sym_outvars, jaxpr.outvars, stage_name,
                                   "end"))
    invars.difference_update(intermediates)
    # handle donation
    num_donated = 0
    if donation_map:
        (invars, outvars,
         num_donated) = _rearrange_in_out_for_donation(invars, outvars,
                                                       donation_map)
    is_donated = [True] * num_donated + [False] * (len(invars) - num_donated)
    jaxpr = Jaxpr(const_dir.keys(), invars, outvars, eqns)
    closed_jaxpr = ClosedJaxpr(jaxpr, const_dir.values())
    return closed_jaxpr, is_donated


def _wrap_by_marker(jaxpr: Jaxpr, name, gensym_fn):
    eqns = []
    new_invars = list(jaxpr.invars)
    new_outvars = list(jaxpr.outvars)
    sym_invars = [gensym_fn(var.aval) for var in new_invars]
    sym_outvars = [gensym_fn(var.aval) for var in new_outvars]
    eqns.append(mark_pipeline_jaxpreqn(new_invars, sym_invars, name, "start"))
    params = dict(name=name,
                  call_jaxpr=Jaxpr([], new_invars + jaxpr.constvars,
                                   new_outvars, jaxpr.eqns))
    eqns.append(
        new_jaxpr_eqn(sym_invars + jaxpr.constvars, sym_outvars, named_call_p,
                      params))
    eqns.append(mark_pipeline_jaxpreqn(sym_outvars, new_outvars, name, "end"))
    return Jaxpr(list(jaxpr.constvars), list(jaxpr.invars), new_outvars, eqns)


def merge_marked_jaxprs_with_named_call(jaxprs: Sequence[ClosedJaxpr],
                                        may_outvars: OrderedSet[Var],
                                        donation_map=None,
                                        prefix=None,
                                        wrap_with_marker=False,
                                        gensym_fn=None) -> ClosedJaxpr:
    """
    Merge continuous jaxprs and remove pipe markers.

    Args:
        jaxprs: jaxprs to be merged.
        may_outvars: outvars of the merged jaxpr.
        donation_map: donation map of merged jaxpr, may have redundant items.
        prefix: name of pipeline marker for merged jaxpr
        insert_hook_after: index of a layer to insert a hook after it.
            The hook records sharding specs of all tensors cross it.
        wrap_with_marker: Whether the returned jaxpr has pipeline marker

    Returns:
        The merged ClosedJaxpr. If insert_hook_after is not None, it returns
        invars of the hook as well.
    """

    def unwrap_with_call(jaxpr, name):
        assert jaxpr.eqns[0].primitive == pipeline_p
        assert jaxpr.eqns[-1].primitive == pipeline_p
        used_var = OrderedSet()
        for eqn in jaxpr.eqns[1:-1]:
            used_var.update([var for var in eqn.invars if isinstance(var, Var)])
        used_var.intersection_update(jaxpr.eqns[0].outvars)
        new_invars = {}
        for invar, outvar in zip(jaxpr.eqns[0].invars, jaxpr.eqns[0].outvars):
            if outvar in used_var:
                new_invars[outvar] = invar
        new_jaxpr = clone_jaxpr(jaxpr, new_invars.keys(), jaxpr.eqns[-1].invars,
                                jaxpr.eqns[1:-1])
        return _wrap_with_call(new_jaxpr, list(new_invars.values()),
                               jaxpr.eqns[-1].outvars, name)

    def has_output(jaxpr):
        return len([v for v in jaxpr.outvars if not isinstance(v, DropVar)])

    name_prefix = prefix or ""
    new_eqns = []
    invars = []
    env = OrderedSet()
    const_dir = {}
    outvars = OrderedSet()
    gensym_fn = gensym_fn or gensym([j.jaxpr for j in jaxprs])
    # Merge everything together
    for i, jaxpr in enumerate(jaxprs):
        const_dir.update(zip(jaxpr.jaxpr.constvars, jaxpr.consts))
        env.update(jaxpr.jaxpr.constvars)
        if has_output(jaxpr.jaxpr):
            call_eqn = unwrap_with_call(jaxpr, name_prefix + str(i))
            new_eqns.append(call_eqn)
            invars.extend(OrderedSet(call_eqn.invars).difference(env))
            env.update(call_eqn.invars + call_eqn.outvars)
        outvars.update(jaxpr.jaxpr.outvars)
    outvars.intersection_update(may_outvars)

    # handle donation
    if donation_map:
        invars, outvars, _ = _rearrange_in_out_for_donation(
            invars, outvars, donation_map)
    # wrap with marker
    jaxpr = Jaxpr(const_dir.keys(), invars, outvars, new_eqns)
    if wrap_with_marker:
        jaxpr = _wrap_by_marker(jaxpr, prefix, gensym_fn)
    closed_jaxpr = ClosedJaxpr(jaxpr, const_dir.values())

    return closed_jaxpr


def create_donation_mapping(initial_mapping, donated_invars, invars, outvars):
    """Infer donation of global invar-outvars."""
    donation_mapping = dict(initial_mapping)
    donated_outvars = OrderedSet()

    for donate, invar in zip(donated_invars, invars):
        if not donate:
            continue
        for outvar in outvars:
            if outvar in donated_outvars:
                continue
            if invar.aval.shape != outvar.aval.shape:
                continue
            donated_outvars.add(outvar)
            donation_mapping[invar] = outvar
            break
        if invar not in donation_mapping:
            logger.warning(
                f"{invar} is marked donated but no match outvar for it")
    return donation_mapping


def get_local_donation_mapping_and_add_missing_invars(computation,
                                                      reversed_donation_mapping,
                                                      gensym_fn):
    """Get the local donation mapping of selected computation and add missing
    input variables of the donated output variables.

    If an outvar is donated from an invar not in the current computation, the
    function add the invar and create a new computation and corresponding to
    the donation mapping.
    """
    invars = OrderedSet(computation.invars)
    donation_mapping = {}
    appended_invars = OrderedSet()
    for var in computation.outvars:
        if var not in reversed_donation_mapping:
            continue
        invar = reversed_donation_mapping[var]
        assert invar.aval.shape == var.aval.shape
        donation_mapping[invar] = var
        if invar not in invars:
            appended_invars.add(invar)
    if not donation_mapping:
        return donation_mapping, computation
    # append invars for donation
    new_invars = list(computation.invars)
    new_outvars = list(computation.outvars)
    new_eqns = list(computation.eqns)
    appended_invars = list(appended_invars)
    if appended_invars:
        new_invars = new_invars + appended_invars
        pipe_start = new_eqns[0]
        new_eqns[0] = mark_pipeline_jaxpreqn(
            pipe_start.invars + appended_invars, pipe_start.outvars +
            list(map(lambda v: gensym_fn(v.aval), appended_invars)),
            pipe_start.params["name"], pipe_start.params["mark_type"])
    # rearrange to keep donated invars and outvars have same index
    new_invars, new_pipe_start = rearrange_vars(new_invars,
                                                list(donation_mapping.keys()),
                                                new_eqns[0], True)
    new_outvars, new_pipe_end = rearrange_vars(new_outvars,
                                               list(donation_mapping.values()),
                                               new_eqns[-1], False)
    new_eqns[0] = new_pipe_start
    new_eqns[-1] = new_pipe_end
    new_computation = JaxPipelineComputation(computation.name, new_invars,
                                             new_outvars, new_eqns,
                                             computation.consts_dir)
    return donation_mapping, new_computation


def split_donate_invars(donation_mapping,
                        stages: Sequence[JaxPipelineComputation], gensym_fn):
    """
    Split donated invars for sliced jaxprs, then rewrite stages.

    Currently, we only donate:
    1. global invars that can be donated(set by users);
    2. buffers for accumulated gradients.
    But if auto-sharding supports, we can add:
    1. local invars not used later in this mesh, not main copy
    2. local invars not used later in all meshes, main copy

    Args:
        donation_mapping (Dict[Var, Var]): known mapping of donations, including
            global invar-outvar and accumulate gradients.
        stages: slices in topology order of execution.

    Returns:
        donate_invars_dict:Sequence[Sequence[bool]]: donate_invars for each
            stage.
    """
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}

    ans = [None for _ in range(len(stages))]
    new_stages = []

    for stage_idx, stage in enumerate(stages):
        # find donation mapping of the stage
        donation_mapping, new_stage = (
            get_local_donation_mapping_and_add_missing_invars(
                stage, reversed_donation_mapping, gensym_fn))
        donated_num = len(donation_mapping)
        ans[stage_idx] = (True,) * donated_num + (False,) * (
            len(new_stage.invars) - donated_num)
        new_stages.append(new_stage)

    return ans, new_stages


def get_donatable_intermediate(stages: Sequence[JaxPipelineComputation],
                               worker_stage_mapping, global_invars):
    """
    Get donatable invars of each stage.

    A donatable invar is:
    1. An intermediate;
    2. Either a main copy never used, or not a main copy.

    Args:
        stages (Sequence[JaxPipelineStage]): all stages.
        worker_stage_mapping (Dict[int, OrderedSet[int]]): indices of stages in
            each mesh.
        global_invars (Sequence[Var] | OrderedSet[Var]): global input variables.

    Returns:
        donatable_list (Sequence[OrderedSet[Var]]): donatable invars of each
            stage.
    """
    global_invars = OrderedSet(global_invars)
    main_copy_at = {}
    stage_at = {}
    for mesh_idx, stage_indices in worker_stage_mapping.items():
        for stage_idx in stage_indices:
            stage = stages[stage_idx]
            for outvar in stage.outvars:
                main_copy_at[outvar] = mesh_idx
            stage_at[stage_idx] = mesh_idx

    donatable_list = []
    used = OrderedSet()
    for stage_idx in reversed(range(len(stages))):
        stage = stages[stage_idx]
        donatable = OrderedSet()
        for invar in stage.invars:
            if invar in global_invars:
                continue  # do not consider global inputs
            if main_copy_at[invar] != stage_at[stage_idx]:
                donatable.add(invar)  # not a main copy
            if invar not in used:
                donatable.add(invar)  # is a main copy never used
        used.update(stage.invars)
        donatable_list.append(donatable)
    donatable_list = list(reversed(donatable_list))
    return donatable_list
