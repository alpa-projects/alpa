"""pipeline stage definitions."""
from abc import ABC, abstractmethod
from copy import copy
from dataclasses import dataclass, field
import itertools as it
import logging
from typing import Sequence, List, Set, Any, Dict

import numpy as np
from jax import jit
from jax._src.util import partial, safe_map
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun, new_jaxpr_eqn, gensym
from jax.interpreters import xla
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.core import (Atom, ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar,
                      gensym, new_jaxpr_eqn, jaxpr_as_fun)

from parax.device_mesh import PhysicalDeviceMesh
from parax.measure_record import StrategyConfig
from parax.pipeline_parallel.primitive_def import (pipeline_p,
                                                   mark_pipeline_jaxpreqn)
from parax.shard_parallel.auto_sharding import (compile_with_search,
                                                compile_with_given_strategy,
                                                get_input_output_sharding_specs)
from parax.util import get_compile_options, jaxpr_to_hlo_computation

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class PipelineStage(ABC):
    """
    Base class of pipeline stages.

    Attributes:
        name (str): The name of the pipeline stage.
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
        """Compile the stage and get the runnable."""
        raise NotImplementedError()


@dataclass
class StrVarPipelineStage:
    """Stringified stage with all Set/Dict have string keys."""

    name: str
    invars: Sequence[str]
    outvars: Sequence[str]

    @classmethod
    def from_pipeline_stage(cls, pipeline_stage: PipelineStage):
        """Construct a StrVarPipelineStage from a PipelineStage."""
        return cls(
            name=pipeline_stage.name,
            invars=[repr(var) for var in pipeline_stage.invars],
            outvars=[repr(var) for var in pipeline_stage.outvars],
        )


@dataclass
class ManualPipelineStage(PipelineStage):
    """
    Base class of manual pipeline stages.

    Attributes:
        pipeline_invars (Set[Var]): The set of input variables receiving from
            the previous pipeline stage.
        global_invars (Set[Var]): The set of input variables from driver
            function inputs.
        local_invars (Set[Var]): The set of input variables from previous
            stages running on the same device.
        pipeline_outvars (Set[Var]): The set of output variables sending to
            the next pipeline stage.
        global_outvars (Set[Var]): The set of output variables that will be
            used as driver function outputs.
        local_outvars (Set[Var]): The set of output variables that will be used
            by future stages running on the same device.
    """

    # invars
    pipeline_invars: Set[Var] = field(default_factory=set)
    global_invars: Set[Var] = field(default_factory=set)
    local_invars: Set[Var] = field(default_factory=set)
    # outvars
    pipeline_outvars: Set[Var] = field(default_factory=set)
    global_outvars: Set[Var] = field(default_factory=set)
    local_outvars: Set[Var] = field(default_factory=set)


@dataclass
class JaxPipelineStage(PipelineStage):
    """
    A pipeline stage defined by Jaxpr.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline stage.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            stage.
    """

    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)

    def closed_jaxpr(self) -> ClosedJaxpr:
        """
        Get the closed Jaxpr of the pipeline stage.

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
        """Return a JIT callable of the pipeline stage."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))


@dataclass
class JaxManualPipelineStage(JaxPipelineStage, ManualPipelineStage):
    pass


@dataclass
class XlaPipelineStage(PipelineStage):
    """A pipeline stage defined by XLA HLO proto."""

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_stage(cls, jax_pipeline_stage: JaxPipelineStage):
        """
        Construct a XlaPipelineStage from a JaxPipelineStage.

        Args:
            jax_pipeline_stage (JaxPipelineStage): the source JaxPipelineStage.
        """
        closed_jaxpr = jax_pipeline_stage.closed_jaxpr()
        backend = xb.get_backend("gpu")
        name = "pipeline_stage_{}".format(jax_pipeline_stage.name)
        built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)

        return cls(
            name=jax_pipeline_stage.name,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            invars=jax_pipeline_stage.invars,
            outvars=jax_pipeline_stage.outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
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
class XlaShardedPipelineStage(PipelineStage):
    """A pipeline stage defined by XLA HLO proto. The XLA HLO is annotated by sharding spec."""

    hlo_proto: Any = None
    donated_invars: Any = None  # TODO(Hao): figure out donated_invars
    strategy_config: StrategyConfig = None
    input_sharding_specs: Any = None
    output_sharding_specs: Any = None

    @classmethod
    def from_auto_sharded_stage(cls,
                                *,
                                jax_pipeline_stage: JaxPipelineStage,
                                auto_sharded_hlo_proto: xc.XlaComputation,
                                strategy_config: StrategyConfig,
                                donated_invars=None):
        # pylint: disable=too-many-locals
        """Run auto-sharding optimizer on a Jax pipeline stage."""
        if not donated_invars:
            donated_invars = (False,) * len(jax_pipeline_stage.invars)
        return cls(
            name=jax_pipeline_stage.name,
            hlo_proto=auto_sharded_hlo_proto,
            strategy_config=strategy_config,
            donated_invars=donated_invars,
            invars=jax_pipeline_stage.invars,
            outvars=jax_pipeline_stage.outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
        from parax.shard_parallel.auto_sharding import HloProtoStatus

        if not isinstance(mesh, PhysicalDeviceMesh):
            raise RuntimeError(
                "Require a pre-allocated physical mesh to compile the runnable."
            )

        strategy_config = self.strategy_config
        logical_mesh_shape = strategy_config.logical_mesh_shape
        xla_computation = xc.XlaComputation(self.hlo_proto)
        backend_name = 'gpu'
        backend = xb.get_backend(backend_name)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        compiled = compile_with_given_strategy(
            backend, xla_computation, self.strategy_config, num_devices,
            mesh.is_distributed, HloProtoStatus.SHARDING_ANNOTATED)
        hlo_module = compiled.hlo_modules()[0]
        if mesh.is_distributed:
            compiled = mesh.compile_remote_executable(
                hlo_module.as_serialized_hlo_module_proto(),
                self.strategy_config, HloProtoStatus.FULLY_OPTIMIZED)

        # Return the final callable
        avals = [var.aval for var in self.invars]
        out_avals = [var.aval for var in self.outvars]
        input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
            hlo_module, num_devices, avals, out_avals, logical_mesh_shape)

        # TODO(Hao): make this better
        self.input_sharding_specs = input_sharding_specs
        self.output_sharding_specs = output_sharding_specs

        return mesh.get_callable_with_arg_handler(compiled, avals, out_avals,
                                                  input_sharding_specs,
                                                  output_sharding_specs,
                                                  self.donated_invars)


def slice_closed_jaxpr_by_manual_pipeline_marks(
    closed_jaxpr: ClosedJaxpr
) -> Sequence[JaxManualPipelineStage]:  # noqa MC0001
    """Slice a Jaxpr into multiple pipeline stages.

    We assume the closed_jaxpr includes pipeline start and end markers. Also,
    the variables in the markers represents the variables being sent
    through the network. While other input variables must be directly from
    the invars.

    Args:
        closed_jaxpr (ClosedJaxpr): the input Jaxpr.

    Returns:
        Sequence[JaxPipelineStage]: A list of sliced pipeline stages.
    """
    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_outvars = set(
        var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    var2stage = {}
    result_stages = []

    current_stage = None
    current_stage_intermediate_vars = set()

    first_eqn = closed_jaxpr.jaxpr.eqns[0]
    assert (first_eqn.primitive is pipeline_p and first_eqn.params["mark_type"] == "start"), \
        "First jaxpr equation must be a pipeline start mark."
    last_eqn = closed_jaxpr.jaxpr.eqns[-1]
    assert (last_eqn.primitive is pipeline_p and last_eqn.params["mark_type"] == "end"), \
        "Last jaxpr equation must be a pipeline end mark."

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = JaxManualPipelineStage(name=eqn.params['name'])
            current_stage_intermediate_vars = set()
            for var in eqn.invars:
                if not isinstance(var, Literal):
                    current_stage.pipeline_invars.add(var)
        assert current_stage is not None

        for var in eqn.invars:
            if isinstance(
                    var, Literal) or (var in current_stage.pipeline_invars) or (
                        var in current_stage_intermediate_vars):
                continue
            if var in global_consts_dir:
                if var not in current_stage.consts_dir:
                    current_stage.consts_dir[var] = global_consts_dir[var]
            elif var in global_invars:
                if var not in current_stage.global_invars:
                    current_stage.global_invars.add(var)
            else:
                if var not in var2stage:
                    raise ValueError("Unknown variable {}".format(var))
                original_stage = var2stage[var]
                if original_stage.name == current_stage.name:
                    if var not in original_stage.local_outvars:
                        original_stage.local_outvars.add(var)
                    if var not in current_stage.local_invars:
                        current_stage.local_invars.add(var)
                else:
                    raise ValueError(
                        "Variable {} should be indicated as a pipeline stage input."
                        .format(var))

        for var in eqn.outvars:
            if not isinstance(var, DropVar):
                current_stage_intermediate_vars.add(var)
                var2stage[var] = current_stage
                if var in global_outvars:
                    current_stage.global_outvars.add(var)

        current_stage.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_stage is not None, "Ending a pipeline stage before its start."
            assert current_stage.name == eqn.params[
                'name'], "Ending a pipeline stage different from its start."
            current_stage.pipeline_outvars = set(
                var for var in eqn.outvars if not isinstance(var, DropVar))
            result_stages.append(current_stage)
            current_stage = None

    for stage in result_stages:
        stage.invars = list(stage.pipeline_invars | stage.global_invars |
                            stage.local_invars)
        stage.outvars = list(stage.pipeline_outvars | stage.global_outvars |
                             stage.local_outvars)

    return result_stages


def get_var_mapping(mapping, var):
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var


def mark_global_and_local_vars(stage: JaxManualPipelineStage,
                               gensym_func) -> JaxPipelineStage:
    """Rewrite pipeline stages so that all inputs and outputs go through the pipeline marker."""
    assert stage.eqns[0].primitive is pipeline_p and stage.eqns[0].params[
        'mark_type'] == 'start'
    assert stage.eqns[-1].primitive is pipeline_p and stage.eqns[-1].params[
        'mark_type'] == 'end'
    new_stage = JaxPipelineStage(stage.name, consts_dir=stage.consts_dir)
    var_alias = {
        var: gensym_func(var.aval)
        for var in it.chain(stage.global_invars, stage.local_invars,
                            stage.global_outvars, stage.local_outvars)
    }

    for eqn in stage.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            # Pipeline start marker
            global_and_local_invars = list(
                it.chain(stage.global_invars, stage.local_invars))
            eqn_invars_without_literal = []
            eqn_outvars_without_literal = []
            for invar, outvar in zip(eqn.invars, eqn.outvars):
                if isinstance(invar, Literal):
                    var_alias[outvar] = invar
                else:
                    eqn_invars_without_literal.append(invar)
                    eqn_outvars_without_literal.append(outvar)
            invars = eqn_invars_without_literal + global_and_local_invars
            outvars = [
                get_var_mapping(var_alias, var)
                for var in eqn_outvars_without_literal + global_and_local_invars
            ]
            new_stage.invars = invars
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            global_and_local_outvars = list(
                it.chain(stage.global_outvars, stage.local_outvars))
            eqn_invars_without_dropvar = []
            eqn_outvars_without_dropvar = []
            for invar, outvar in zip(eqn.invars, eqn.outvars):
                if not isinstance(outvar, DropVar):
                    eqn_invars_without_dropvar.append(invar)
                    eqn_outvars_without_dropvar.append(outvar)
            invars = [
                get_var_mapping(var_alias, var)
                for var in eqn_invars_without_dropvar + global_and_local_outvars
            ]
            outvars = eqn_outvars_without_dropvar + global_and_local_outvars
            new_stage.outvars = outvars
        else:
            invars = [get_var_mapping(var_alias, var) for var in eqn.invars]
            outvars = [get_var_mapping(var_alias, var) for var in eqn.outvars]
        new_stage.eqns.append(eqn._replace(invars=invars, outvars=outvars))

    return new_stage


def slice_eqns_by_pipeline_marks(closed_jaxpr: ClosedJaxpr):
    sliced_eqns = []
    current_stage_eqns = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage_eqns is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage_eqns = []
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_stage_eqns is not None, "Ending a pipeline stage before its start."
            sliced_eqns.append(current_stage_eqns)
            current_stage_eqns = None
        else:
            assert current_stage_eqns is not None
            current_stage_eqns.append(eqn)
    assert current_stage_eqns is None
    return sliced_eqns


def add_pipeline_marks_for_sliced_eqns(closed_jaxpr: ClosedJaxpr, sliced_eqns):
    n_layers = len(sliced_eqns)
    layer_pipeline_invars = [set() for _ in range(n_layers)]
    layer_pipeline_outvars = [set() for _ in range(n_layers)]
    var_layer_dict = {}

    for var in closed_jaxpr.jaxpr.invars:
        var_layer_dict[var] = -1

    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if not isinstance(var, Literal) and var_layer_dict[var] != i:
                    layer_pipeline_invars[i].add(var)
                    if var_layer_dict[var] != -1:
                        layer_pipeline_outvars[var_layer_dict[var]].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i

    for var in closed_jaxpr.jaxpr.outvars:
        if not isinstance(var, Literal) and var_layer_dict[var] != -1:
            layer_pipeline_outvars[var_layer_dict[var]].add(var)

    gensym_func = gensym([closed_jaxpr.jaxpr])
    var_mapping = {}

    new_eqns = []
    for i, eqns in enumerate(sliced_eqns):
        # pipeline start eqn
        stage_var_mapping = {}

        pipeline_start_invars = []
        pipeline_start_outvars = []
        for var in layer_pipeline_invars[i]:
            new_var = gensym_func(var.aval)
            pipeline_start_invars.append(get_var_mapping(var_mapping, var))
            pipeline_start_outvars.append(new_var)
            stage_var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_start_invars,
                                   pipeline_start_outvars, str(i), 'start'))
        # all other eqns
        for eqn in eqns:
            new_invars = [
                get_var_mapping(stage_var_mapping, var) for var in eqn.invars
            ]
            new_eqns.append(
                new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.source_info))
        # pipeline end eqn
        pipeline_end_invars = []
        pipeline_end_outvars = []
        for var in layer_pipeline_outvars[i]:
            new_var = gensym_func(var.aval)
            pipeline_end_invars.append(get_var_mapping(stage_var_mapping, var))
            pipeline_end_outvars.append(new_var)
            var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars,
                                   str(i), 'end'))
    new_jaxpr = Jaxpr(
        closed_jaxpr.jaxpr.constvars,
        closed_jaxpr.jaxpr.invars,
        [
            get_var_mapping(var_mapping, var)
            for var in closed_jaxpr.jaxpr.outvars
        ],
        new_eqns,
    )
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
    return new_closed_jaxpr


def slice_closed_jaxpr_by_full_pipeline_marks(
        closed_jaxpr: ClosedJaxpr) -> Sequence[JaxPipelineStage]:  # noqa MC0001
    global_consts_dir = dict(
        zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))

    result_stages = []
    current_stage = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = JaxPipelineStage(name=eqn.params['name'])
            for var in eqn.invars:
                if isinstance(var, Literal):
                    pass
                elif var in global_consts_dir:
                    current_stage.consts_dir[var] = global_consts_dir[var]
                else:
                    current_stage.invars.append(var)

        assert current_stage is not None
        current_stage.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_stage is not None, "Ending a pipeline stage before its start."
            assert current_stage.name == eqn.params[
                'name'], "Ending a pipeline stage different from its start."
            for var in eqn.outvars:
                current_stage.outvars.append(var)
            result_stages.append(current_stage)
            current_stage = None

    return result_stages


def mark_missing_vars_in_pipeline_marks(stages: Sequence[JaxPipelineStage],
                                        global_invars, global_outvars):
    gensym_func = gensym([stage.closed_jaxpr().jaxpr for stage in stages])
    var_stage_id = {}
    for var in global_invars:
        if not isinstance(var, Literal):
            var_stage_id[var] = -1

    stage_additional_invars = [set() for _ in stages]
    stage_additional_outvars = [set() for _ in stages]
    for i, stage in enumerate(stages):
        for eqn in stage.eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in stage.consts_dir and
                        var not in stage.invars):
                    source_stage_id = var_stage_id[var]
                    if source_stage_id != i:
                        if (source_stage_id != -1 and
                                var not in stages[source_stage_id].outvars):
                            stage_additional_outvars[source_stage_id].add(var)
                        stage_additional_invars[i].add(var)
            for var in eqn.outvars:
                var_stage_id[var] = i

    for var in global_outvars:
        source_stage_id = var_stage_id[var]
        if source_stage_id != -1 and var not in stages[source_stage_id].outvars:
            stage_additional_outvars[source_stage_id].add(var)

    new_stages = []

    for i, stage in enumerate(stages):
        assert stage.eqns[0].primitive is pipeline_p and stage.eqns[0].params[
            'mark_type'] == 'start'
        assert stage.eqns[-1].primitive is pipeline_p and stage.eqns[-1].params[
            'mark_type'] == 'end'
        new_stage = JaxPipelineStage(stage.name, consts_dir=stage.consts_dir)

        stage_var_mapping = {
            var: gensym_func(var.aval)
            for var in stage_additional_invars[i] | stage_additional_outvars[i]
        }
        pipeline_start_invars = list(stage.eqns[0].invars)
        pipeline_start_outvars = [
            get_var_mapping(stage_var_mapping, var)
            for var in stage.eqns[0].outvars
        ]
        new_stage.invars = list(stage.invars)
        for var in stage_additional_invars[i]:
            pipeline_start_invars.append(var)
            pipeline_start_outvars.append(stage_var_mapping[var])
        pipeline_start_invars_without_literal = []
        pipeline_start_outvars_without_literal = []
        for invar, outvar in zip(pipeline_start_invars, pipeline_start_outvars):
            if isinstance(invar, Literal):
                stage_var_mapping[outvar] = invar
            else:
                pipeline_start_invars_without_literal.append(invar)
                pipeline_start_outvars_without_literal.append(outvar)
        new_stage.invars = list(pipeline_start_invars_without_literal)
        new_stage.eqns.append(stage.eqns[0]._replace(
            invars=pipeline_start_invars_without_literal,
            outvars=pipeline_start_outvars_without_literal))

        for eqn in stage.eqns[1:-1]:
            new_stage.eqns.append(
                eqn._replace(invars=[
                    get_var_mapping(stage_var_mapping, var)
                    for var in eqn.invars
                ],
                             outvars=[
                                 get_var_mapping(stage_var_mapping, var)
                                 for var in eqn.outvars
                             ]))

        pipeline_end_invars = [
            get_var_mapping(stage_var_mapping, var)
            for var in stage.eqns[-1].invars
        ]
        pipeline_end_outvars = list(stage.eqns[-1].outvars)
        for var in stage_additional_outvars[i]:
            pipeline_end_invars.append(stage_var_mapping[var])
            pipeline_end_outvars.append(var)
        pipeline_end_invars_without_dropvar = []
        pipeline_end_outvars_without_dropvar = []
        for invar, outvar in zip(pipeline_end_invars, pipeline_end_outvars):
            if not isinstance(outvar, DropVar):
                pipeline_end_invars_without_dropvar.append(invar)
                pipeline_end_outvars_without_dropvar.append(outvar)
        new_stage.outvars = list(pipeline_end_outvars_without_dropvar)
        new_stage.eqns.append(stage.eqns[-1]._replace(
            invars=pipeline_end_invars_without_dropvar,
            outvars=pipeline_end_outvars_without_dropvar))
        new_stages.append(new_stage)

    return new_stages


def generate_sharded_xla_stages(name: str,
                                jax_stages: Sequence[JaxPipelineStage],
                                physical_mesh, logical_mesh_choices,
                                logical_mesh_search_mode,
                                memory_budget_per_device, search_task,
                                record_file):
    """Generate sharded XLA stages by running the sharding optimizer given JaxPipleStages."""
    invars = set()
    outvars = set()
    eqns = []
    consts_dir = {}
    for stage in jax_stages:
        consts_dir.update(stage.consts_dir)
        # Do not add local invars into the invars
        invars.update([var for var in stage.invars if var not in outvars])
        outvars.update(stage.outvars)
        eqns += stage.eqns
    jaxpr = Jaxpr(
        constvars=list(consts_dir.keys()),
        invars=list(invars),
        outvars=list(outvars),
        eqns=eqns,
    )
    closed_jaxpr = ClosedJaxpr(jaxpr, consts_dir.values())
    backend_name = 'gpu'
    backend = xb.get_backend(backend_name)
    built = jaxpr_to_hlo_computation(name, closed_jaxpr, None, backend)
    stage_protos, strategy_config = compile_with_search(
        backend,
        built,
        physical_mesh,
        logical_mesh_choices,
        logical_mesh_search_mode,
        memory_budget_per_device,
        search_task,
        record_file,
        multiple_stages=True)
    stages = [
        XlaShardedPipelineStage.from_auto_sharded_stage(
            auto_sharded_hlo_proto=proto,
            jax_pipeline_stage=stage,
            strategy_config=strategy_config)
        for stage, proto in zip(jax_stages, stage_protos)
    ]
    return stages


def mark_grad_mesh(invars: Sequence[Var], stages: Sequence[JaxPipelineStage], stage_to_mesh):
    # TODO(yonghao): now assume all gradients are variables(not literal)
    outvar2mesh = {}
    for stage in stages:
        for var in stage.outvars:
            if isinstance(var, Var):
                outvar2mesh[var] = stage_to_mesh[stages]
    return {
        invar: outvar2mesh[invar] for invar in invars if invar in outvar2mesh
    }


def slice_apply_gradient(closed_jaxpr: ClosedJaxpr, grad_mesh: Dict[Var, int]):
    """
    Slice the apply gradient jaxpr based on mesh allocation information
    Args:
        closed_jaxpr(ClosedJaxpr): closed jaxpr of apply_gradient function. 
        grad_mesh(Dict[Var, int]): dict indicating which mesh the variable is at. 
        If not in the dict, the variable should be a global parameter.
    Returns:
        jaxprs(List[ClosedJaxpr]): The i-th ClosedJaxpr runs at the i-th cluster.
        infered_global_invars(Dict[Var, List[int]]): Indicating which clusters each
        input variable of apply_gradient function should be sent to.
    """

    def add_allocation(cur: Set, add: Set):
        if cur is None:
            return add
        else:
            return cur.union(add)

    global_invars = closed_jaxpr.jaxpr.invars
    mesh_num = max(set(grad_mesh.values())) + 1  # + 1 for 0-base
    eqn_mesh = dict()
    var_mesh = {var: set([mesh]) for var, mesh in grad_mesh.items()}
    infered_global_invars = dict()
    constvars = [list() for _ in range(mesh_num)]
    consts = [list() for _ in range(mesh_num)]
    sliced_eqns = [list() for _ in range(mesh_num)]
    invars = [list() for _ in range(mesh_num)]
    outvars = [list() for _ in range(mesh_num)]
    # propagate mesh assignments from input
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        at_mesh = None
        for invar in eqn.invars:
            if isinstance(invar, Var) and invar in var_mesh:
                at_mesh = add_allocation(at_mesh, var_mesh[invar])
        if at_mesh is not None:
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    cur_mesh = var_mesh[invar] if invar in var_mesh else None
                    var_mesh[invar] = add_allocation(cur_mesh, at_mesh)
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    var_mesh[outvar] = at_mesh
            eqn_mesh[eqn_idx] = at_mesh
    # propagate back
    for reversed_idx, eqn in enumerate(reversed(closed_jaxpr.eqns)):
        eqn_idx = len(closed_jaxpr.eqns) - 1 - reversed_idx
        if eqn_idx not in eqn_mesh:
            at_mesh = None
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar) and outvar in var_mesh:
                    at_mesh = add_allocation(at_mesh, var_mesh[outvar])
            if at_mesh is not None:
                eqn_mesh[eqn_idx] = at_mesh
                for invar in eqn.invars:
                    if isinstance(invar, Var):
                        cur_mesh = var_mesh[invar] if invar in var_mesh else None
                        var_mesh[invar] = add_allocation(cur_mesh, at_mesh)
    for eqn_idx, eqn in enumerate(closed_jaxpr.eqns):
        if eqn_idx in eqn_mesh:
            for mesh in eqn_mesh[eqn_idx]:
                sliced_eqns[mesh].append(eqn)
        else:
            # all inputs are infered, all outputs are not assigned
            sliced_eqns[0].append(eqn)
            for invar in eqn.invars:
                if isinstance(invar, Var):
                    if invar not in var_mesh:
                        var_mesh[invar] = [0]
            for outvar in eqn.outvars:
                if not isinstance(outvar, DropVar):
                    assert (outvar not in var_mesh or
                            (len(var_mesh[outvar]) == 1 and
                             var_mesh[outvar][0] == 0))
                    var_mesh[outvar] = [0]
    # grouping invars and outvars
    for invar in global_invars:
        assert invar in var_mesh
        for mesh in var_mesh[invar]:
            invars[mesh].append(invar)
        infered_global_invars[invar] = var_mesh[invar]
    for outvar in closed_jaxpr.jaxpr.outvars:
        assert outvar in var_mesh
        for mesh in var_mesh[outvar]:
            outvars[mesh].append(outvar)
    # grouping consts and constvars
    for aval, var in zip(closed_jaxpr.consts, closed_jaxpr.jaxpr.constvars):
        assert var in var_mesh
        for mesh in var_mesh[var]:
            consts[mesh].append(aval)
            constvars[mesh].append(var)

    jaxprs = [
        ClosedJaxpr(Jaxpr(constvars[i], invars[i], outvars[i], sliced_eqns[i]),
                    consts[i]) for i in range(mesh_num)
    ]
    return jaxprs, infered_global_invars


def get_gradient(compute_jaxpr: ClosedJaxpr, slices: Sequence[ClosedJaxpr]):
    gradients = [
        outvar for outvar in compute_jaxpr.jaxpr.outvars
        if isinstance(outvar, Var)
    ]
    gradients = set(gradients)
    is_gradients = [[
        isinstance(outvar, Var) and outvar in gradients
        for outvar in slice.jaxpr.outvars
    ]
                    for slice in slices]
    return is_gradients


def is_pipeline(closed_jaxpr: ClosedJaxpr, is_gradient):
    """If the closed_jaxpr is a pipeline stage, ignore the last equation(pipeline end marker)"""
    from parax.pipeline_parallel.primitive_def import pipeline_p
    last_eqn = closed_jaxpr.eqns[-1]
    if last_eqn.primitive is not pipeline_p:
        return False, closed_jaxpr.jaxpr.outvars, closed_jaxpr.eqns, is_gradient
    new_is_gradient = set()
    for i, outvar in enumerate(last_eqn.outvars):
        if outvar in is_gradient:
            new_is_gradient.add(last_eqn.invars[i])
    return True, last_eqn.invars, closed_jaxpr.eqns[:-1], new_is_gradient


def _compute_to_acc(closed_jaxpr: ClosedJaxpr, donated_invars, is_gradient):
    """
    Transform compute_grad jaxpr to accumulate_grad jaxpr.
    Args:
        closed_jaxpr: the compute_grad jaxpr. It is supposed to be the one in a single pipeline-stage
        donated_invars: donated invars of the input jaxpr
        is_gradient: a set of vars accumulated in outputs
    """
    from jax.lax import add_p
    # FIXME(yonghao): add division of microbatch num
    pipeline, old_outs, old_eqns, is_gradient = is_pipeline(
        closed_jaxpr, is_gradient)
    gensym_fn = gensym([
        closed_jaxpr.jaxpr,
    ])
    # add input:
    grad_invars = {
        outvar: gensym_fn(outvar.aval)
        for outvar in old_outs
        if isinstance(outvar) and outvar in is_gradient
    }
    new_invars = closed_jaxpr.jaxpr.invars + list(grad_invars.values())
    # modify output
    new_outvars = [
        outvar if not isinstance(outvar, Var) or outvar not in is_gradient else
        gensym_fn(outvar.aval) for outvar in old_outs
    ]
    # add grad acc eqns
    acc_grad_eqns = []
    for i, outvar in enumerate(old_outs):
        if outvar in is_gradient:
            acc_grad_eqns.append(
                new_jaxpr_eqn([outvar, grad_invars[outvar]], [new_outvars[i]],
                              add_p, {}))
    new_eqns = old_eqns + acc_grad_eqns
    if pipeline:
        # replace the input of old pipeline-end marker
        old_pipeline_eqn = closed_jaxpr.eqns[-1]
        new_eqns.append(
            new_jaxpr_eqn(new_outvars, old_pipeline_eqn.outvars,
                          old_pipeline_eqn.primitive, old_pipeline_eqn.params,
                          old_pipeline_eqn.source_info))
        new_outvars = old_pipeline_eqn.outvars
    # construct jaxpr
    jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, new_invars, new_outvars,
                  new_eqns)
    # add donate invars
    donated_invars = tuple(donated_invars) + (True,) * len(grad_invars)
    # return closed_jaxpr and new donate_invars
    return ClosedJaxpr(jaxpr, closed_jaxpr.consts), donated_invars


def compute_to_acc(raw_jaxpr, slices, donated_invars_set):
    is_gradient_sets = get_gradient(raw_jaxpr, slices)
    new_slices = [
        _compute_to_acc(slice, donated_invars, is_gradient)
        for slice, donated_invars, is_gradient in zip(
            slices, donated_invars_set, is_gradient_sets)
    ]
    return new_slices


def compute_to_acc_pipe(compute_jaxpr: ClosedJaxpr):
    from jax.lax import add_p
    # Assume that no grad is literal
    raw_gradients = [
        outvar for outvar in compute_jaxpr.jaxpr.outvars
        if isinstance(outvar, Var)
    ]
    assert len(raw_gradients) == len(compute_jaxpr.jaxpr.outvars)
    # from raw_gradients to gradients(cross pipeline marker)
    gradients = {}
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            for i, outvar in enumerate(eqn.outvars):
                if outvar in raw_gradients and outvar not in gradients:
                    gradients[outvar] = eqn.invars[i]
    for outvar in raw_gradients:
        assert outvar in gradients, 'all gradients are captured by pipeline marker'
    gradients = list(gradients.values())
    # generate new variables
    gensym_fn = gensym([compute_jaxpr])
    grad_invars = {outvar: gensym_fn(outvar.aval) for outvar in gradients}
    grad_outs = {outvar: gensym_fn(outvar.aval) for outvar in gradients}
    new_glob_invars = compute_jaxpr.jaxpr.invars + list(grad_invars.values())
    # modify output, here all grads are acc_grad
    new_glob_outvars = [
        grad_outs[outvar]
        if isinstance(outvar, Var) and outvar in gradients else outvar
        for outvar in compute_jaxpr.jaxpr.outvars
    ]
    gradients = set(gradients)
    # rewrite eqns
    # TODO(yonghao): clear unused outputs(grad not used later)
    new_eqns = []
    pipe_start = None
    pipe_eqns = []
    to_acc = []
    for eqn in compute_jaxpr.eqns:
        if eqn.primitive is pipeline_p:
            if eqn.params['type'] == 'start':
                pipe_start = eqn
                continue
            if eqn.params['type'] == 'end':
                # add grad used in this stage in pipeline start
                grad_in_after_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                grad_out_before_pipe = {
                    outvar: gensym_fn(outvar.aval) for outvar in to_acc
                }
                new_pipe_start = mark_pipeline_jaxpreqn(
                    pipe_start.invars + map(lambda x: grad_invars[x], to_acc),
                    pipe_start.outvars +
                    map(lambda x: grad_in_after_pipe[x], to_acc),
                    pipe_start.params['name'], pipe_start.params['mark_type'])
                new_eqns.append(new_pipe_start)
                # add normal eqns
                new_eqns.extend(pipe_eqns)
                # add acc grad(adds)
                for gradient in to_acc:
                    new_eqns.append(
                        new_jaxpr_eqn([grad_in_after_pipe[gradient], gradient],
                                      [grad_out_before_pipe[gradient]], add_p,
                                      {}))
                # add grad created in this stage in pipeline end
                new_pipe_end = mark_pipeline_jaxpreqn(
                    eqn.invars + map(lambda x: grad_out_before_pipe[x], to_acc),
                    eqn.outvars + map(lambda x: grad_outs[x], to_acc),
                    eqn.params['name'], eqn.params['mark_type'])
                new_eqns.append(new_pipe_end)
                pipe_start = None
                pipe_eqns = []
                to_acc = []
                continue
        pipe_eqns.append(eqn)
        for outvar in eqn.outvars:
            if not isinstance(outvar, DropVar) and outvar in gradients:
                # collect gradients in this stage
                to_acc.append(outvar)
    new_jaxpr = Jaxpr(compute_jaxpr.jaxpr.constvars, new_glob_invars,
                      new_glob_outvars, new_eqns)
    # We do not modify donate_invars here, as it is only to append Trues.
    # Instead we return grad_outs for runtime
    return ClosedJaxpr(new_jaxpr, compute_jaxpr.consts), grad_outs


def rewrite_apply_grad(closed_jaxpr: ClosedJaxpr, in_dict, barrier: JaxprEqn):
    # rewrite the apply_gradients by replacing all inputs with info from in_dict
    mapping = {
        barrier_out: in_dict[barrier_in]
        for barrier_in, barrier_out in zip(barrier.invars, barrier.outvars)
    }

    def map_var(var):
        if isinstance(var, Var) and var in mapping:
            return mapping[var]
        return var

    new_glob_invars = [mapping[var] for var in closed_jaxpr.jaxpr.invars]
    new_eqns = []
    for eqn in closed_jaxpr.eqns:
        new_invars = [map_var(var) for var in eqn.invars]
        new_eqns.append(
            new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive, eqn.params))
    new_jaxpr = Jaxpr(closed_jaxpr.jaxpr.constvars, new_glob_invars,
                      closed_jaxpr.jaxpr.outvars, new_eqns)
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)