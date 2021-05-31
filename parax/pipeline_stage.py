"""pipeline stage definitions."""
import itertools as it
from copy import copy
from dataclasses import dataclass, field
from typing import Sequence, List, Set, Any, Dict

from abc import ABC, abstractmethod
from jax import jit
from jax._src.util import partial, safe_map, extend_name_stack, wrap_name
from jax.core import Atom, Var, JaxprEqn, Jaxpr, ClosedJaxpr, jaxpr_as_fun
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.interpreters import xla

from parax.pipeline_primitive_def import pipeline_p

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


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
    def get_runnable(self):
        """Compile the stage and get the runnable."""
        raise NotImplementedError()


@dataclass
class JaxPipelineStage(PipelineStage):
    """
    Pipeline stage with JaxPr.

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

    def get_runnable(self):
        """Return a JIT callable of the pipeline stage."""
        closed_jaxpr = self.closed_jaxpr()
        return jit(jaxpr_as_fun(closed_jaxpr))


def build_hlo_computation_from_jaxpr(name, closed_jaxpr):
    in_avals = [var.aval for var in closed_jaxpr.jaxpr.invars]
    consts = closed_jaxpr.consts
    map(xla.prefetch, it.chain(consts, xla.jaxpr_literals(closed_jaxpr.jaxpr)))

    backend = 'gpu'
    tuple_args = len(in_avals) > 100  # pass long arg lists as tuple for TPU

    c = xb.make_computation_builder("pipeline_stage_{}".format(name))
    xla_consts = xla._xla_consts(c, consts)
    xla_args, _ = xla._xla_callable_args(c, in_avals, tuple_args, donated_invars=None)
    axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
    out_nodes = xla.jaxpr_subcomp(
        c, closed_jaxpr.jaxpr, backend, axis_env, xla_consts,
        extend_name_stack(wrap_name(name, 'stage')), *xla_args)
    out_tuple = xc.ops.Tuple(c, out_nodes)
    built = c.build(out_tuple)
    return built


@dataclass
class XlaPipelineStage(PipelineStage):
    """
    Pipeline stage with XLA HLO protos.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline stage.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            stage.
    """

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_stage(cls, jax_pipeline_stage: JaxPipelineStage):
        """
        Construct a XlaPipelineStage from a JaxPipelineStage.

        Args:
            jax_pipeline_stage (JaxPipelineStage): the source JaxPipelineStage.
        """
        print("=" * 80)
        closed_jaxpr = jax_pipeline_stage.closed_jaxpr()
        built = build_hlo_computation_from_jaxpr(jax_pipeline_stage.name, closed_jaxpr)
        print("built", built.as_hlo_text())

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

    def get_runnable(self):
        """Return a callable of the pipeline stage."""
        out_avals = [var.aval for var in self.outvars]
        xla_computation = xc.XlaComputation(self.hlo_proto)
        tuple_args = len(self.invars) > 100  # pass long arg lists as tuple for TPU
        nreps = 1
        backend = 'gpu'
        backend = xb.get_backend(backend)
        device = backend.get_default_device_assignment(1)[0]
        options = xb.get_compile_options(
            num_replicas=nreps,
            num_partitions=1,
            device_assignment=(device.id,) if device else None)
        options.parameter_is_tupled_arguments = tuple_args
        compiled = backend.compile(xla_computation, compile_options=options)
        result_handlers = map(partial(xla.aval_to_result_handler, device), out_avals)
        kept_var_idx = range(len(self.invars))
        return partial(xla._execute_compiled, compiled, out_avals, result_handlers, kept_var_idx)


def mark_global_and_local_vars(stage: JaxPipelineStage, gensym_func):
    """Rewrite pipeline stages so that all inputs and outputs go through the pipeline marker"""
    assert stage.eqns[0].primitive is pipeline_p and stage.eqns[0].params['mark_type'] == 'start'
    assert stage.eqns[-1].primitive is pipeline_p and stage.eqns[-1].params['mark_type'] == 'end'
    new_stage = copy(stage)
    new_stage.eqns = []
    var_alias = {var: gensym_func(var.aval) for var in it.chain(
        stage.global_invars, stage.local_invars, stage.global_outvars,
        stage.local_outvars)}

    for eqn in stage.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            # Pipeline start marker
            global_and_local_invars = list(it.chain(stage.global_invars, stage.local_invars))
            invars = eqn.invars + global_and_local_invars
            outvars = eqn.outvars + [var_alias[var] for var in global_and_local_invars]
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            global_and_local_outvars = list(it.chain(stage.global_outvars, stage.local_outvars))
            invars = eqn.invars + [var_alias[var] for var in global_and_local_outvars]
            outvars = eqn.outvars + global_and_local_outvars
        else:
            invars = [var_alias.get(var, var) if isinstance(var, Var) else var for var in eqn.invars]
            outvars = [var_alias.get(var, var) if isinstance(var, Var) else var for var in eqn.outvars]
        new_stage.eqns.append(eqn._replace(invars=invars, outvars=outvars))

    return new_stage


def generate_sharded_xla_stages(name: str, jax_stages: Sequence[JaxPipelineStage]):
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
    print("=" * 40 + " closed_jaxpr " + name + " " + "=" * 40)
    print(closed_jaxpr)
    hlo_computation = build_hlo_computation_from_jaxpr(name, closed_jaxpr)
    print("=" * 40 + " hlo_computation " + name + " " + "=" * 40)
    print(hlo_computation.as_hlo_text())
    pass

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
