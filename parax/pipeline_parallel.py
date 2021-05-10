"""pipeline parallel on a single device."""
from typing import Sequence, Set

import jax
from jax import linear_util as lu
from jax._src.util import safe_map
from jax.core import Var, DropVar, ClosedJaxpr, Literal
from jax.interpreters import partial_eval as pe

from parax.pipeline_primitive_def import pipeline_p
from parax.pipeline_stage import PipelineStage, JaxPipelineStage, XlaPipelineStage
from parax.pipe import JaxPipeline

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


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

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = JaxPipelineStage(name=eqn.params['name'])
            for var in eqn.invars:
                if not isinstance(var, Literal):
                    current_stage.pipeline_invars.add(var)
        assert current_stage is not None

        for var in eqn.invars:
            if isinstance(var, Literal) or (var in current_stage.pipeline_invars) or (
                    var in current_stage.intermediate_vars):
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
    """
    Single-node local pipeline runner.

    Args:
        name (str): pipeline runner name.
        global_invals (List[DeviceArrays]): input device arrays to the stage.

    Returns:
        None
    """

    def __init__(self, name, global_invals):
        self.name = name
        self.env = {}
        self.global_invals = global_invals

    def run_stage(self, stage: PipelineStage, prev_stage_pipeline_outvals: Set[Var] = None):
        """
        Run a pipeline stage.

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
        outvals = dict(zip(stage.outvars, outvals_list))
        local_outvals = {var: outvals[var] for var in stage.local_outvars}
        pipeline_outvals = {var: outvals[var] for var in stage.pipeline_outvars}
        global_outvals = {var: outvals[var] for var in stage.global_outvars}
        self.env.update(local_outvals)
        return pipeline_outvals, global_outvals


def local_pipeline_runtime(pipeline_stages: Sequence[PipelineStage], global_invars: Sequence[Var],
                           global_outvars: Sequence[Var]):
    """
    Return a callable that runs all pipeline stages on a single local device.

    Args:
        pipeline_stages (Sequence[PipelineStage]): the pipeline stages to be
            executed.
        global_invars (Sequence[Var]): Global input variables.
        global_outvars (Sequence[Var]): Global output variables.

    Returns:
        ret_func (function): the returned function.
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
    """Pipeline parallel callable."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    xla_pipeline_stages = [XlaPipelineStage.from_jax_pipeline_stage(stage)
                           for stage in jax_pipeline_stages]
    return local_pipeline_runtime(xla_pipeline_stages, global_invars, global_outvars)


@lu.cache
def distributed_pipeline_parallel_callable(
        fun: lu.WrappedFun,
        *avals
):
    """Distributed pipeline parallel callable."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    xla_pipeline_stages = [XlaPipelineStage.from_jax_pipeline_stage(stage)
                           for stage in jax_pipeline_stages]
    jp = JaxPipeline(pipeline_stages=xla_pipeline_stages,
                     global_invars=global_invars,
                     global_outvars=global_outvars)
    return lambda *args, **kwargs: jp.run(*args, **kwargs)  # pylint: disable=unnecessary-lambda
