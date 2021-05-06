"""pipeline parallel on a single device"""
import itertools as it
from dataclasses import dataclass, field
from typing import Sequence, List, Set, Any, Dict
from abc import ABC, abstractmethod

import numpy as np

import jax
from jax import jit
from jax import linear_util as lu
from jax.core import Atom, Var, DropVar, JaxprEqn, Jaxpr, ClosedJaxpr, Primitive, Literal, abstract_unit, jaxpr_as_fun
from jax.interpreters import xla, ad, partial_eval as pe
from jax.lib import xla_client as xc, xla_bridge as xb
from jax._src.util import partial, safe_map, extend_name_stack, wrap_name

unsafe_map, map = map, safe_map  # type: ignore

# Define a Jax primitive to mark start/end of a pipeline stage.
pipeline_p = Primitive('pipeline')
pipeline_p.multiple_results = True

def mark_pipeline(*args, name: str, mark_type: str):
    """Mark the start/end of a pipeline stage.

    Args:
        *args: represents the pipeline input/output of a pipeline stage.
        name (str): Name of the pipeline stage.
        mark_type (str): start or end of a pipeline stage, can be "start",
            "end", "jvp_start", or "jvp_end". The latter two are used for
            backward pass.
    """
    if mark_type not in ('start', 'end', 'jvp_start', 'jvp_end'):
        raise ValueError('Unknown mark type: %s' % mark_type)
    return pipeline_p.bind(*args, name=name, mark_type=mark_type)

def _pipeline_impl(*args, **kwargs):
    # The pipeline marker acts as an identity function.
    return args if len(args) > 0 else (None, )

def _pipeline_abstract_eval(*args, **kwargs):
    return args if len(args) > 0 else (abstract_unit, )

def _pipeline_xla_translation(c, *args, **kwargs):
    return xc.ops.Tuple(c, args) if len(args) > 0 else xc.ops.Tuple(c, (xc.ops.Constant(c, np.float32(0.0)), ))

def _pipeline_value_and_jvp(arg_values, arg_tangents, name, mark_type):
    primal_outs = mark_pipeline(*arg_values, name=name, mark_type=mark_type)
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type == "start" or mark_type == "jvp_start":
        tangent_mark_type = "jvp_start"
    elif mark_type == "end" or mark_type == "jvp_end":
        tangent_mark_type = "jvp_end"
    else:
        raise ValueError("Invalid mark_type")
    tangent_outs = mark_pipeline(*arg_tangents, name=name, mark_type=tangent_mark_type)
    return primal_outs, tangent_outs

def _pipeline_transpose(ct, *args, name, mark_type):
    # TODO(zhuohan): Check the semantics here works for higher order gradients.
    if mark_type == "start" or mark_type == "jvp_start":
        transposed_mark_type = "end"
    elif mark_type == "end" or mark_type == "jvp_end":
        transposed_mark_type = "start"
    else:
        raise ValueError("Invalid mark_type")
    res = mark_pipeline(*ct, name=name, mark_type=transposed_mark_type)
    return res

pipeline_p.def_impl(_pipeline_impl)
pipeline_p.def_abstract_eval(_pipeline_abstract_eval)
xla.translations[pipeline_p] = _pipeline_xla_translation
ad.primitive_jvps[pipeline_p] = _pipeline_value_and_jvp
ad.primitive_transposes[pipeline_p] = _pipeline_transpose


@dataclass
class PipelineStage(ABC):
    """Base class of pipeline stages.

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
    # intermediate vars
    intermediate_vars: Set[Var] = field(default_factory=set)

    @abstractmethod
    def get_runnable(self):
        pass


@dataclass
class JaxPipelineStage(PipelineStage):
    """Pipeline stage with JaxPr.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline stage.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            stage.
    """

    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)

    def closed_jaxpr(self) -> ClosedJaxpr:
        """Get the closed Jaxpr of the pipeline stage.

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


@dataclass
class XlaPipelineStage(PipelineStage):
    """Pipeline stage with XLA HLO protos.

    Attributes:
        eqns (List[JaxprEqn]): Jaxpr equations of the pipeline stage.
        consts_dir: Dict[Atom, Any]: All the constants used in the pipeline
            stage.
    """

    hlo_proto: bytes = field(default_factory=b"")

    @classmethod
    def from_jax_pipeline_stage(cls, jax_pipeline_stage: JaxPipelineStage):
        """Construct a XlaPipelineStage from a JaxPipelineStage.

        Args:
            jax_pipeline_stage (JaxPipelineStage): the source JaxPipelineStage.
        """
        closed_jaxpr = jax_pipeline_stage.closed_jaxpr()
        in_avals = [var.aval for var in jax_pipeline_stage.invars]
        consts = closed_jaxpr.consts
        map(xla.prefetch, it.chain(consts, xla.jaxpr_literals(closed_jaxpr.jaxpr)))

        backend = 'gpu'
        tuple_args = len(in_avals) > 100  # pass long arg lists as tuple for TPU

        c = xb.make_computation_builder("pipeline_stage_{}".format(jax_pipeline_stage.name))
        xla_consts = xla._xla_consts(c, consts)
        xla_args, donated_invars = xla._xla_callable_args(c, in_avals, tuple_args, donated_invars=None)
        axis_env = xla.AxisEnv(nreps=1, names=(), sizes=())  # All named axes have been vmapped
        out_nodes = xla.jaxpr_subcomp(
            c, closed_jaxpr.jaxpr, backend, axis_env, xla_consts,
            extend_name_stack(wrap_name(jax_pipeline_stage.name, 'stage')), *xla_args)
        out_tuple = xc.ops.Tuple(c, out_nodes)
        built = c.build(out_tuple)

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
        return partial(xla._execute_compiled, compiled, out_avals, result_handlers)


def slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr: ClosedJaxpr) -> Sequence[JaxPipelineStage]:
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

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = JaxPipelineStage(name=eqn.params['name'])
            for var in eqn.invars:
                if not isinstance(var, Literal):
                    current_stage.pipeline_invars.add(var)
        assert current_stage is not None

        for var in eqn.invars:
            if isinstance(var, Literal) or (var in current_stage.pipeline_invars) or (var in current_stage.intermediate_vars):
                continue
            elif var in global_consts_dir:
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
                current_stage.intermediate_vars.add(var)
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


class LocalPipelineRunner:
    def __init__(self, name, global_invals):
        self.name = name
        self.env = {}
        self.global_invals = global_invals

    def run_stage(self, stage: PipelineStage, prev_stage_pipeline_outvals: Set[Var]=None):
        """Run a pipeline stage.

        Args:
            stage (PipelineStage): The pipeline stage to run.
            prev_stage_pipeline_outvals (Set[Var], optional): Results from
                the previous pipeline stage. When set to None, we'll fetch
                the values of pipeline input variables from global input
                variables, which is used for the very first pipeline stage.

        Returns:
            Two dictionaries with values of pipeline & global output variables.
        """
        runnable = stage.get_runnable()
        invals_list = []
        for var in stage.invars:
            if var in stage.pipeline_invars:
                if prev_stage_pipeline_outvals is None:
                    invals_list.append(self.global_invals[var])
                else:
                    invals_list.append(prev_stage_pipeline_outvals[var])
            elif var in stage.global_invars:
                invals_list.append(self.global_invals[var])
            else:
                assert var in stage.local_invars
                invals_list.append(self.env[var])
        outvals_list = runnable(*invals_list)
        outvals = {var: val for var, val in zip(stage.outvars, outvals_list)}
        local_outvals = {var: outvals[var] for var in stage.local_outvars}
        pipeline_outvals = {var: outvals[var] for var in stage.pipeline_outvars}
        global_outvals = {var: outvals[var] for var in stage.global_outvars}
        self.env.update(local_outvals)
        return pipeline_outvals, global_outvals


def local_pipeline_runtime(pipeline_stages: Sequence[PipelineStage], global_invars: Sequence[Var], global_outvars: Sequence[Var]):
    """Return a callable that runs all pipeline stages on a single local device.

    Args:
        pipeline_stages (Sequence[PipelineStage]): the pipeline stages to be
            executed.
        global_invars (Sequence[Var]): Global input variables.
        global_outvars (Sequence[Var]): Global output variables.
    """
    def ret_func(*args, **kwargs):
        assert not kwargs, "kwargs not supported"
        global_invals = dict(zip(global_invars, args))
        global_outvals = {}
        runners = {}
        pipeline_outvals = None
        for stage in pipeline_stages:
            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(stage.name, global_invals)
            pipeline_outvals, stage_global_outvals = runners[stage.name].run_stage(stage, pipeline_outvals)
            global_outvals.update(stage_global_outvals)
        global_outvals_list = []
        for var in global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                global_outvals_list.append(global_outvals[var])
        return global_outvals_list

    return ret_func


@lu.cache
def pipeline_parallel_callable(
    fun: lu.WrappedFun,
    *avals
):
    with jax.disable_jit():
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    xla_pipeline_stages = [XlaPipelineStage.from_jax_pipeline_stage(stage) for stage in jax_pipeline_stages]
    return local_pipeline_runtime(xla_pipeline_stages, global_invars, global_outvars)
