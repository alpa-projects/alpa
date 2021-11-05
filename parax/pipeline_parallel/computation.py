"""Pipeline computation definitions."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from typing import Sequence, Set, Any, Dict

from jax import jit
from jax._src.util import partial, safe_map
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun, new_jaxpr_eqn, gensym
from jax.interpreters import xla
from jax.lib import xla_bridge as xb, xla_client as xc
import numpy as np

from parax.device_mesh import PhysicalDeviceMesh
from parax.measure_record import StrategyConfig
from parax.mesh_executable import PartialGradAccMeshDriverExecutable
from parax.pipeline_parallel.primitive_def import (pipeline_p,
                                                   mark_pipeline_jaxpreqn)
from parax.pipeline_parallel.manual_layer_slicing import get_var_mapping
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                get_input_output_sharding_specs,
                                                HloProtoStatus)
from parax.util import get_compile_options, jaxpr_to_hlo_computation, setup_computation_alias, log_jaxpr

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
                   consts_dir={
                       k: v for k, v in zip(closed_jaxpr.jaxpr.constvars,
                                            closed_jaxpr.consts)
                   })


@dataclass
class XlaPipelineComputation(PipelineComputation):
    """A pipeline computation defined by XLA HLO proto."""

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_computation(
            cls, jax_pipeline_computation: JaxPipelineComputation):
        """
        Construct a XlaPipelineComputation from a JaxPipelineComputation.

        Args:
            jax_pipeline_computation (JaxPipelineComputation): the source JaxPipelineComputation.
        """
        closed_jaxpr = jax_pipeline_computation.closed_jaxpr()
        backend = xb.get_backend("gpu")
        name = "pipeline_computation_{}".format(jax_pipeline_computation.name)
        built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)

        return cls(
            name=jax_pipeline_computation.name,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            invars=jax_pipeline_computation.invars,
            outvars=jax_pipeline_computation.outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""
        out_avals = [var.aval for var in self.outvars]
        xla_computation = xc.XlaComputation(self.hlo_proto)
        tuple_args = len(
            self.invars) > 100  # pass long arg lists as tuple for TPU
        backend = 'gpu'
        backend = xb.get_backend(backend)
        device = backend.get_default_device_assignment(1)[0]
        options = get_compile_options(
            num_replicas=1,
            num_partitions=1,
            device_assignment=(device.id,) if device else None,
            use_spmd_partitioning=False,
            parameter_is_tupled_arguments=tuple_args,
            build_random_seed=42,
        )

        compiled = backend.compile(xla_computation, compile_options=options)
        result_handlers = map(partial(xla.aval_to_result_handler, device),
                              out_avals)
        kept_var_idx = range(len(self.invars))
        return partial(xla._execute_compiled, compiled, out_avals,
                       result_handlers, kept_var_idx)


@dataclass
class XlaShardedPipelineComputation(PipelineComputation):
    """A pipeline computation defined by XLA HLO proto. The XLA HLO is annotated by sharding spec."""

    hlo_proto: Any = None
    donated_invars: Any = None
    strategy_config: StrategyConfig = None
    input_sharding_specs: Any = None
    output_sharding_specs: Any = None
    output_acc_grad_indices: Sequence[int] = None

    @classmethod
    def from_auto_sharded_computation(
        cls,
        *,
        jax_pipeline_computation: JaxPipelineComputation,
        auto_sharded_hlo_proto: xc.XlaComputation,
        strategy_config: StrategyConfig,
        donated_invars=None,
        acc_grad_outvars=set()):
        # pylint: disable=too-many-locals
        """Run auto-sharding optimizer on a Jax pipeline computation."""
        if not donated_invars:
            donated_invars = (False,) * len(jax_pipeline_computation.invars)

        acc_grad_indices = [
            out_idx
            for out_idx, outvar in enumerate(jax_pipeline_computation.outvars)
            if outvar in acc_grad_outvars
        ]

        return cls(name=jax_pipeline_computation.name,
                   hlo_proto=auto_sharded_hlo_proto,
                   strategy_config=strategy_config,
                   donated_invars=donated_invars,
                   invars=jax_pipeline_computation.invars,
                   outvars=jax_pipeline_computation.outvars,
                   output_acc_grad_indices=acc_grad_indices)

    def get_compiled(self, mesh=None):

        if not isinstance(mesh, PhysicalDeviceMesh):
            raise RuntimeError(
                "Require a pre-allocated physical mesh to compile the runnable."
            )

        strategy_config = self.strategy_config
        logical_mesh_shape = strategy_config.logical_mesh_shape
        xla_computation = xc.XlaComputation(self.hlo_proto)
        setup_computation_alias(xla_computation, self.donated_invars)
        backend_name = 'gpu'
        backend = xb.get_backend(backend_name)
        num_devices = np.prod(logical_mesh_shape)
        rewrite_for_grad_acc = len(self.output_acc_grad_indices) > 0
        compiled = compile_with_given_strategy(
            backend,
            xla_computation,
            self.strategy_config,
            num_devices,
            mesh.is_distributed,
            HloProtoStatus.SHARDING_ANNOTATED,
            rewrite_for_grad_acc=rewrite_for_grad_acc,
            rewrite_grad_acc_indices=self.output_acc_grad_indices)

        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
            compiled.hlo_modules()[0], num_devices, avals, out_avals,
            strategy_config.logical_mesh_shape)
        self.input_sharding_specs = input_sharding_specs
        self.output_sharding_specs = output_sharding_specs
        return compiled

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline computation."""

        compiled = self.get_compiled(mesh)
        hlo_module = compiled.hlo_modules()[0]

        # Return the final callable
        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        mesh_executable = PartialGradAccMeshDriverExecutable(
            mesh, compiled, self.strategy_config, avals, out_avals,
            self.donated_invars, self.output_acc_grad_indices)

        return mesh_executable.get_driver_callable()

    def hlo_proto_str(self):
        xla_computation = xc.XlaComputation(self.hlo_proto)
        return xla_computation.as_hlo_text()


def slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr: ClosedJaxpr) -> Sequence[JaxPipelineComputation]:
    """Slice a closed jaxpr into multiple JaxPipelineComputation by full pipeline markers."""
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_computations = []
    current_computation = None

    log_jaxpr(closed_jaxpr, "new_jaxpr")

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_computation is None, "Defining a pipeline computation inside a pipeline computation is not allowed."
            current_computation = JaxPipelineComputation(
                name=eqn.params['name'])
            for var in eqn.invars:
                if isinstance(var, Literal):
                    pass
                elif var in global_consts_dir:
                    current_computation.consts_dir[var] = global_consts_dir[var]
                else:
                    current_computation.invars.append(var)

        assert current_computation is not None
        current_computation.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_computation is not None, "Ending a pipeline computation before its start."
            assert current_computation.name == eqn.params[
                'name'], "Ending a pipeline computation different from its start."
            for var in eqn.outvars:
                current_computation.outvars.append(var)
            result_computations.append(current_computation)
            current_computation = None

    return result_computations


def mark_missing_vars_in_pipeline_marks(
        computations: Sequence[JaxPipelineComputation], global_invars,
        global_outvars):
    """Add missing global vars in pipeline markers."""
    gensym_func = gensym(
        [computation.closed_jaxpr().jaxpr for computation in computations])
    var_computation_id = {}
    for var in global_invars:
        if not isinstance(var, Literal):
            var_computation_id[var] = -1

    computation_additional_invars = [set() for _ in computations]
    computation_additional_outvars = [set() for _ in computations]
    for i, computation in enumerate(computations):
        for eqn in computation.eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in computation.consts_dir and
                        var not in computation.invars):
                    source_computation_id = var_computation_id[var]
                    if source_computation_id != i:
                        if (source_computation_id != -1 and var not in
                                computations[source_computation_id].outvars):
                            computation_additional_outvars[
                                source_computation_id].add(var)
                        computation_additional_invars[i].add(var)
            for var in eqn.outvars:
                var_computation_id[var] = i

    for var in global_outvars:
        source_computation_id = var_computation_id[var]
        if source_computation_id != -1 and var not in computations[
                source_computation_id].outvars:
            computation_additional_outvars[source_computation_id].add(var)

    new_computations = []

    for i, computation in enumerate(computations):
        assert computation.eqns[0].primitive is pipeline_p and computation.eqns[
            0].params['mark_type'] == 'start'
        assert computation.eqns[-1].primitive is pipeline_p and computation.eqns[
            -1].params['mark_type'] == 'end'
        new_computation = JaxPipelineComputation(
            computation.name, consts_dir=computation.consts_dir)

        computation_var_mapping = {
            var: gensym_func(var.aval)
            for var in computation_additional_invars[i] |
            computation_additional_outvars[i]
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
            new_computation.eqns.append(
                eqn._replace(invars=[
                    get_var_mapping(computation_var_mapping, var)
                    for var in eqn.invars
                ],
                             outvars=[
                                 get_var_mapping(computation_var_mapping, var)
                                 for var in eqn.outvars
                             ]))

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
    clear unused vars cross pipeline computations.
    mainly to remove grad and only keep accumulated grad
    """

    def dce_pipe_marker(marker: JaxprEqn, used_set):
        kept_indices = [
            i for i, var in enumerate(marker.outvars) if var in used_set
        ]
        new_marker = mark_pipeline_jaxpreqn(
            [marker.invars[i] for i in kept_indices],
            [marker.outvars[i] for i in kept_indices], marker.params['name'],
            marker.params['mark_type'])
        return new_marker

    global_used = set(global_outvars)
    new_computations = []
    for computation in reversed(jax_pipeline_computations):
        new_eqns = []
        # handle pipe end
        pipe_end = computation.eqns[-1]
        assert (pipe_end.primitive is pipeline_p and
                pipe_end.params['mark_type']
                == 'end'), 'computation not ended by a pipeline marker'
        new_pipe_end = dce_pipe_marker(pipe_end, global_used)
        new_eqns.append(new_pipe_end)
        # handle normal instructions
        local_used = set(new_pipe_end.invars)
        for eqn in reversed(computation.eqns[1:-1]):
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar) and outvar in local_used:
                    new_eqns.append(eqn)
                    local_used.update([
                        invar for invar in eqn.invars if isinstance(invar, Var)
                    ])
        # handle pipe start
        pipe_start = computation.eqns[0]
        assert (pipe_start.primitive is pipeline_p and
                pipe_start.params['mark_type']
                == 'start'), 'computation not started by a pipeline marker'
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


def rearrange_vars(vars,
                   selected: Sequence[Var],
                   pipe_marker=None,
                   is_input=True):
    """
    Rearrange vars to let those in selected be the first. If the pipe_marker is given,
    rearrange invars and outvars in pipemarker also.

    Args:
        vars (Sequence[Var]): all vars to be rearranged.
        selected (Sequence[Var]): vars selected to be prior.
        pipe_marker (JaxprEqn): pipe marker corresponding to vars
        is_input (bool): the var is input of pipe_marker, if False, it is output
    """
    new_vars = list(selected)
    selected = set(selected)
    for var in vars:
        if var not in selected:
            new_vars.append(var)

    if pipe_marker is None:
        return new_vars

    if is_input:
        new_invars = new_vars
        invar_idx = {v: idx for idx, v in enumerate(pipe_marker.invars)}
        new_outvars = [
            pipe_marker.outvars[invar_idx[var]] for var in new_invars
        ]
    else:
        new_outvars = new_vars
        outvar_idx = {v: idx for idx, v in enumerate(pipe_marker.outvars)}
        new_invars = [
            pipe_marker.invars[outvar_idx[var]] for var in new_outvars
        ]
    new_marker = mark_pipeline_jaxpreqn(new_invars, new_outvars,
                                        pipe_marker.params['name'],
                                        pipe_marker.params['mark_type'])
    return new_vars, new_marker


def generate_sharded_xla_computations(
        name: str, jax_computations: Sequence[JaxPipelineComputation],
        computation_donate_invars, physical_mesh, logical_mesh_choices,
        logical_mesh_search_mode, memory_budget_per_device, acc_grad_outvars,
        search_task, record_file):
    """Generate sharded XLA computations by running the auto-sharding pass 
    on the given JaxPipelineComputations."""
    invars = set()
    outvars = set()
    donation_mapping = dict()
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
    backend_name = 'gpu'
    backend = xb.get_backend(backend_name)
    built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)
    computation_protos, strategy_config = compile_with_search(
        backend,
        built,
        invars,
        outvars,
        dummy_donated_invars,
        physical_mesh,
        logical_mesh_choices,
        logical_mesh_search_mode,
        memory_budget_per_device,
        search_task,
        record_file,
        multiple_stages=True,
        grad_acc_num_micro_batches=None,
        bypass_device_assignment_check=physical_mesh.is_distributed)
    computations = [
        XlaShardedPipelineComputation.from_auto_sharded_computation(
            auto_sharded_hlo_proto=proto,
            jax_pipeline_computation=computation,
            strategy_config=strategy_config,
            donated_invars=donate_invars,
            acc_grad_outvars=acc_grad_outvars)
        for computation, proto, donate_invars in zip(
            jax_computations, computation_protos, computation_donate_invars)
    ]
    return computations


def merge_computation_jaxprs(jaxprs: Sequence[ClosedJaxpr],
                             used: Set[Var],
                             new_marker_name,
                             donation_mapping=None) -> ClosedJaxpr:
    """
    Merge continuous jaxprs and remove pipe markers.:

    Args:
        jaxprs (Sequence[ClosedJaxpr]): jaxprs to be merged
        used (Set[Var]): out variables used later
        new_marker_name (str): name of merged pipeline used in marker
        donation_mapping (Dict[Var, Var]): donation mapping of merged jaxpr, may have redundant items
    """
    new_invars = dict()
    new_outvars = dict()
    new_eqns = []
    var_map = dict()

    # handle const vars:
    new_constvars = dict()
    for jaxpr in jaxprs:
        new_constvars.update(dict(zip(jaxpr.jaxpr.constvars, jaxpr.consts)))

    for idx, jaxpr in enumerate(jaxprs):
        # handle pipeline start marker:
        pipe_start = jaxpr.eqns[0]
        for invar, outvar in zip(pipe_start.invars, pipe_start.outvars):
            if invar not in var_map:
                # is not local output, the outvar is kept
                if invar in new_constvars:
                    continue
                # is already set in earlier computations
                if invar in new_invars:
                    var_map[outvar] = new_invars[invar]
                    continue
                new_invars[invar] = outvar
            else:
                # is local output, the outvar is redirected
                var_map[outvar] = var_map[invar]
        # handle normal eqns
        for eqn in jaxpr.eqns[1:-1]:
            new_local_invars = [get_var_mapping(var_map, v) for v in eqn.invars]
            new_eqns.append(
                new_jaxpr_eqn(new_local_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.source_info))
        # handle pipeline end marker
        pipe_end = jaxpr.eqns[-1]
        for invar, outvar in zip(pipe_end.invars, pipe_end.outvars):
            if outvar in used:
                new_outvars[outvar] = get_var_mapping(var_map, invar)
            var_map[outvar] = get_var_mapping(var_map, invar)

    new_pipe_start = mark_pipeline_jaxpreqn(list(new_invars.keys()),
                                            list(new_invars.values()),
                                            new_marker_name, 'start')
    new_pipe_end = mark_pipeline_jaxpreqn(list(new_outvars.values()),
                                          list(new_outvars.keys()),
                                          new_marker_name, 'end')
    new_eqns = [new_pipe_start] + new_eqns + [new_pipe_end]
    constvars = set(new_constvars.keys())
    new_invars = [k for k in new_invars.keys() if k not in constvars]
    new_outvars = list(new_outvars.keys())
    if donation_mapping:
        new_invars_set = set(new_invars)
        new_outvars_set = set(new_outvars)
        donation_mapping = {
            k: v
            for k, v in donation_mapping.items()
            if k in new_invars_set and v in new_outvars_set
        }
        new_invars = rearrange_vars(new_invars, donation_mapping.keys())
        new_outvars = rearrange_vars(new_outvars, donation_mapping.values())
    return ClosedJaxpr(
        Jaxpr(list(new_constvars.keys()), new_invars, new_outvars, new_eqns),
        list(new_constvars.values()))


def create_donation_mapping(initial_mapping, donated_invars, invars, outvars):
    """Infer donation of global invar-outvars."""
    donation_mapping = initial_mapping
    donated_outvars = set()

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
                f"{invar} is marked as donated but actually no match outvar")
    return donation_mapping


def get_donation_mapping_and_modify(stage, reversed_donation_mapping,
                                    gensym_fn):
    invars = set(stage.invars)
    donation_mapping = dict()
    appended_invars = set()
    for var in stage.outvars:
        if var not in reversed_donation_mapping:
            continue
        invar = reversed_donation_mapping[var]
        assert invar.aval.shape == var.aval.shape
        donation_mapping[invar] = var
        if invar not in invars:
            appended_invars.add(invar)
    if not donation_mapping:
        return donation_mapping, stage
    # append invars for donation
    new_invars = list(stage.invars)
    new_outvars = list(stage.outvars)
    new_eqns = list(stage.eqns)
    appended_invars = list(appended_invars)
    if appended_invars:
        new_invars = new_invars + appended_invars
        pipe_start = new_eqns[0]
        new_eqns[0] = mark_pipeline_jaxpreqn(
            pipe_start.invars + appended_invars, pipe_start.outvars +
            list(map(lambda v: gensym_fn(v.aval), appended_invars)),
            pipe_start.params['name'], pipe_start.params['mark_type'])
    # rearrange to keep donated invars and outvars have same index
    new_invars, new_pipe_start = rearrange_vars(new_invars,
                                                list(donation_mapping.keys()),
                                                new_eqns[0], True)
    new_outvars, new_pipe_end = rearrange_vars(new_outvars,
                                               list(donation_mapping.values()),
                                               new_eqns[-1], False)
    new_eqns[0] = new_pipe_start
    new_eqns[-1] = new_pipe_end
    new_stage = JaxPipelineComputation(stage.name, new_invars, new_outvars,
                                       new_eqns, stage.consts_dir)
    return donation_mapping, new_stage


def split_donate_invars(donation_mapping,
                        stages: Sequence[JaxPipelineComputation]):
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
            global invar-outvar and accumulate gradients
        stages: slices in topology order of execution.
    Returns:
        donate_invars_dict:Sequence[Sequence[bool]]: donate_invars for each stage
    """
    reversed_donation_mapping = {v: k for k, v in donation_mapping.items()}
    gensym_fn = gensym([stage.closed_jaxpr().jaxpr for stage in stages])
    # global last use to consider if the main copy can be discarded

    ans = [None for _ in range(len(stages))]
    new_stages = []

    for stage_idx, stage in enumerate(stages):
        # find donation mapping of the stage
        donation_mapping, new_stage = get_donation_mapping_and_modify(
            stage, reversed_donation_mapping, gensym_fn)
        donated_num = len(donation_mapping)
        ans[stage_idx] = (True,) * donated_num + (False,) * (
            len(new_stage.invars) - donated_num)
        new_stages.append(new_stage)

    return ans, new_stages
