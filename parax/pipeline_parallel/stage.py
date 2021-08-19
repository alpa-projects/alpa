"""pipeline stage definitions."""
import itertools as it
import logging
from dataclasses import dataclass, field
from typing import Sequence, List, Set, Any, Dict
from abc import ABC, abstractmethod
from copy import copy

import numpy as np
from jax import jit
from jax._src.util import partial, safe_map
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun
from jax.interpreters import xla
from jax.lib import xla_bridge as xb, xla_client as xc

# pylint: disable=redefined-builtin
from parax.auto_sharding import compile_with_search, compile_with_given_strategy, get_input_output_sharding_specs
from parax.device_mesh import PhysicalDeviceMesh
from parax.measure_record import StrategyConfig
from parax.pipeline_parallel.primitive_def import pipeline_p
from parax.util import get_compile_options, jaxpr_to_hlo_computation

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
        pipeline_invars (Set[Var]): The set of input variables receiving from
            the previous pipeline stage.
        global_invars (Set[Var]): The set of input variables from driver
            function inputs.
        local_invars (Set[Var]): The set of input variables from previous
            stages running on the same device.
        outvars (Sequence[Var]): The list of output variables, corresponding to
            the order of the runnable outputs.
        pipeline_outvars (Set[Var]): The set of output variables sending to
            the next pipeline stage.
        global_outvars (Set[Var]): The set of output variables that will be
            used as driver function outputs.
        local_outvars (Set[Var]): The set of output variables that will be used
            by future stages running on the same device.
    """

    name: str
    # invars
    invars: Sequence[Var] = field(default_factory=list)
    pipeline_invars: Set[Var] = field(default_factory=set)
    global_invars: Set[Var] = field(default_factory=set)
    local_invars: Set[Var] = field(default_factory=set)
    # outvars
    outvars: Sequence[Var] = field(default_factory=list)
    pipeline_outvars: Set[Var] = field(default_factory=set)
    global_outvars: Set[Var] = field(default_factory=set)
    local_outvars: Set[Var] = field(default_factory=set)

    @abstractmethod
    def get_runnable(self, mesh=None):
        """Compile the stage and get the runnable."""
        raise NotImplementedError()


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
            constvars=self.consts_dir.keys(),
            invars=self.invars,
            outvars=self.outvars,
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, self.consts_dir.values())
        return closed_jaxpr

    def get_runnable(self, mesh=None):
        """Return a JIT callable of the pipeline stage."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))


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
        built = jaxpr_to_hlo_computation(jax_pipeline_stage.name, closed_jaxpr)
        #print("=" * 80)
        #print("built", built.as_hlo_text())

        return cls(
            name=jax_pipeline_stage.name,
            hlo_proto=built.as_serialized_hlo_module_proto(),
            invars=jax_pipeline_stage.invars,
            pipeline_invars=jax_pipeline_stage.pipeline_invars,
            global_invars=jax_pipeline_stage.global_invars,
            local_invars=jax_pipeline_stage.local_invars,
            outvars=jax_pipeline_stage.outvars,
            pipeline_outvars=jax_pipeline_stage.pipeline_outvars,
            global_outvars=jax_pipeline_stage.global_outvars,
            local_outvars=jax_pipeline_stage.local_outvars,
        )

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
        out_avals = [var.aval for var in self.outvars]
        xla_computation = xc.XlaComputation(self.hlo_proto)
        tuple_args = len(self.invars) > 100  # pass long arg lists as tuple for TPU
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
        result_handlers = map(partial(xla.aval_to_result_handler, device), out_avals)
        kept_var_idx = range(len(self.invars))
        return partial(xla._execute_compiled, compiled, out_avals, result_handlers, kept_var_idx)


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
            donated_invars = (False, ) * len(jax_pipeline_stage.invars)
        return cls(
            name=jax_pipeline_stage.name,
            hlo_proto=auto_sharded_hlo_proto,
            strategy_config=strategy_config,
            donated_invars=donated_invars,
            invars=jax_pipeline_stage.invars,
            pipeline_invars=jax_pipeline_stage.pipeline_invars,
            global_invars=jax_pipeline_stage.global_invars,
            local_invars=jax_pipeline_stage.local_invars,
            outvars=jax_pipeline_stage.outvars,
            pipeline_outvars=jax_pipeline_stage.pipeline_outvars,
            global_outvars=jax_pipeline_stage.global_outvars,
            local_outvars=jax_pipeline_stage.local_outvars,
        )

    # def input_sharding_specs_on_mesh(self, logical_mesh):
    #     """Return the input sharding spec on a given logical mesh."""
    #     if not isinstance(logical_mesh, LogicalDeviceMesh):
    #         raise RuntimeError("Require a logical mesh to obtain the input sharding spec.")
    #     avals = [var.aval for var in self.invars]
    #     input_shardings = self.hlo_module.spmd_parameters_shardings()
    #     input_sharding_specs = [hlo_sharding_to_sharding_spec(proto_tuple, aval, logical_mesh)
    #                             for (proto_tuple, aval) in zip(input_shardings, avals)]
    #     return input_sharding_specs
    #
    # def output_sharding_specs_on_mesh(self, logical_mesh):
    #     """Return the output sharding spec on a given logical mesh."""
    #     if not isinstance(logical_mesh, LogicalDeviceMesh):
    #         raise RuntimeError("Require a logical mesh to obtain the input sharding spec.")
    #     out_avals = [var.aval for var in self.outvars]
    #     output_sharding = self.hlo_module.spmd_output_sharding()
    #     output_sharding_specs = hlo_sharding_to_sharding_spec(output_sharding, out_avals, logical_mesh)
    #     return output_sharding_specs
    #
    # def input_output_sharding_spec_on_mesh(self, logical_mesh):
    #     """Return the input and output sharding spec on a given logical mesh."""
    #     if not isinstance(logical_mesh, LogicalDeviceMesh):
    #         raise RuntimeError("Require a logical mesh to obtain the input sharding spec.")
    #     avals = [var.aval for var in self.invars]
    #     out_avals = [var.aval for var in self.outvars]
    #     input_sharding_specs, output_sharding_specs = get_input_output_sharding_specs(
    #         hlo_module, num_devices, avals, out_avals, logical_mesh_shape)

    def get_runnable(self, mesh=None):
        """Return a callable of the pipeline stage."""
        from parax.auto_sharding import HloProtoStatus

        if not isinstance(mesh, PhysicalDeviceMesh):
            raise RuntimeError("Require a pre-allocated physical mesh to compile the runnable.")

        strategy_config = self.strategy_config
        logical_mesh_shape = strategy_config.logical_mesh_shape
        xla_computation = xc.XlaComputation(self.hlo_proto)
        backend_name = 'gpu'
        backend = xb.get_backend(backend_name)
        num_devices = np.prod(strategy_config.logical_mesh_shape)
        compiled = compile_with_given_strategy(
                backend, xla_computation, self.strategy_config,
                num_devices, mesh.is_distributed, HloProtoStatus.SHARDING_ANNOTATED)
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
                                                  input_sharding_specs, output_sharding_specs,
                                                  self.donated_invars)


def slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr: ClosedJaxpr) -> Sequence[JaxPipelineStage]: # noqa MC0001
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
    global_outvars = set(var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts_dir = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    var2stage = {}
    result_stages = []

    current_stage = None
    current_stage_intermediate_vars = set()

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = JaxPipelineStage(name=eqn.params['name'])
            current_stage_intermediate_vars = set()
            for var in eqn.invars:
                if not isinstance(var, Literal):
                    current_stage.pipeline_invars.add(var)
        assert current_stage is not None

        for var in eqn.invars:
            if isinstance(var, Literal) or (var in current_stage.pipeline_invars) or (
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
                    raise ValueError("Variable {} should be indicated as a pipeline stage input.".format(var))

        for var in eqn.outvars:
            if not isinstance(var, DropVar):
                current_stage_intermediate_vars.add(var)
                var2stage[var] = current_stage
                if var in global_outvars:
                    current_stage.global_outvars.add(var)

        current_stage.eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_stage is not None, "Ending a pipeline stage before its start."
            assert current_stage.name == eqn.params['name'], "Ending a pipeline stage different from its start."
            current_stage.pipeline_outvars = set(var for var in eqn.outvars if not isinstance(var, DropVar))
            result_stages.append(current_stage)
            current_stage = None

    for stage in result_stages:
        stage.invars = list(stage.pipeline_invars | stage.global_invars | stage.local_invars)
        stage.outvars = list(stage.pipeline_outvars | stage.global_outvars | stage.local_outvars)

    return result_stages


def mark_global_and_local_vars(stage: JaxPipelineStage, gensym_func):
    """Rewrite pipeline stages so that all inputs and outputs go through the pipeline marker."""
    assert stage.eqns[0].primitive is pipeline_p and stage.eqns[0].params['mark_type'] == 'start'
    assert stage.eqns[-1].primitive is pipeline_p and stage.eqns[-1].params['mark_type'] == 'end'
    new_stage = copy(stage)
    new_stage.eqns = []
    var_alias = {var: gensym_func(var.aval) for var in it.chain(
        stage.global_invars, stage.local_invars, stage.global_outvars,
        stage.local_outvars)}

    def get_alias(var):
        if isinstance(var, Var) and var in var_alias:
            return var_alias[var]
        else:
            return var

    for eqn in stage.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            # Pipeline start marker
            global_and_local_invars = list(it.chain(stage.global_invars, stage.local_invars))
            eqn_invars_without_literal = []
            eqn_outvars_without_literal = []
            for invar, outvar in zip(eqn.invars, eqn.outvars):
                if isinstance(invar, Literal):
                    var_alias[outvar] = invar
                else:
                    eqn_invars_without_literal.append(invar)
                    eqn_outvars_without_literal.append(outvar)
            invars = eqn_invars_without_literal + global_and_local_invars
            outvars = [get_alias(var) for var in eqn_outvars_without_literal + global_and_local_invars]
            new_stage.invars = invars
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            global_and_local_outvars = list(it.chain(stage.global_outvars, stage.local_outvars))
            eqn_invars_without_dropvar = []
            eqn_outvars_without_dropvar = []
            for invar, outvar in zip(eqn.invars, eqn.outvars):
                if not isinstance(outvar, DropVar):
                    eqn_invars_without_dropvar.append(invar)
                    eqn_outvars_without_dropvar.append(outvar)
            invars = [get_alias(var) for var in eqn_invars_without_dropvar + global_and_local_outvars]
            outvars = eqn_outvars_without_dropvar + global_and_local_outvars
            new_stage.outvars = outvars
        else:
            invars = [get_alias(var) for var in eqn.invars]
            outvars = [get_alias(var) for var in eqn.outvars]
        new_stage.eqns.append(eqn._replace(invars=invars, outvars=outvars))

    return new_stage


def generate_sharded_xla_stages(name: str, jax_stages: Sequence[JaxPipelineStage], physical_mesh,
                                logical_mesh_choices, logical_mesh_search_mode,
                                memory_budget_per_device, search_task, record_file):
    """Generate sharded XLA stages by running the sharding optimizer given JaxPipleStages."""
    invars = set()
    outvars = set()
    eqns = []
    consts_dir = {}
    for stage in jax_stages:
        consts_dir.update(stage.consts_dir)
        invars.update(stage.global_invars, stage.pipeline_invars)
        outvars.update(stage.global_outvars, stage.pipeline_outvars)
        eqns += stage.eqns
    jaxpr = Jaxpr(
        constvars=consts_dir.keys(),
        invars=invars,
        outvars=outvars,
        eqns=eqns,
    )
    closed_jaxpr = ClosedJaxpr(jaxpr, consts_dir.values())
    backend_name = 'gpu'
    backend = xb.get_backend(backend_name)
    built_computation = jaxpr_to_hlo_computation(name, closed_jaxpr, backend_name=backend_name)
    stage_protos, strategy_config = compile_with_search(
        backend, built_computation, physical_mesh, logical_mesh_choices,
        logical_mesh_search_mode, memory_budget_per_device, search_task, record_file,
        multiple_stages=True)
    stages = [XlaShardedPipelineStage.from_auto_sharded_stage(auto_sharded_hlo_proto=proto, jax_pipeline_stage=stage,
                                                              strategy_config=strategy_config)
              for stage, proto in zip(jax_stages, stage_protos)]
    return stages


@dataclass
class StrVarPipelineStage:
    """Stringified stage with all Set/Dict have string keys."""

    name: str
    # invars
    invars: Sequence[str]
    pipeline_invars: Set[str]
    global_invars: Set[str]
    local_invars: Set[str]
    # outvars
    outvars: Sequence[str]
    pipeline_outvars: Set[str]
    global_outvars: Set[str]
    local_outvars: Set[str]

    @classmethod
    def from_pipeline_stage(cls, pipeline_stage: PipelineStage):
        """Construct a StrVarPipelineStage from a PipelineStage."""
        return cls(
            name=pipeline_stage.name,
            invars=[repr(var) for var in pipeline_stage.invars],
            pipeline_invars={repr(var) for var in pipeline_stage.pipeline_invars},
            global_invars={repr(var) for var in pipeline_stage.global_invars},
            local_invars={repr(var) for var in pipeline_stage.local_invars},
            outvars=[repr(var) for var in pipeline_stage.outvars],
            pipeline_outvars={repr(var) for var in pipeline_stage.pipeline_outvars},
            global_outvars={repr(var) for var in pipeline_stage.global_outvars},
            local_outvars={repr(var) for var in pipeline_stage.local_outvars},
        )
