"""pipeline parallel on a single device."""
from typing import Sequence, Set, Mapping, Any

import jax
from jax import linear_util as lu
from jax._src.util import safe_map
from jax.core import Var, DropVar, ClosedJaxpr, Literal, gensym
from jax.interpreters import partial_eval as pe

from parax.pipeline_primitive_def import pipeline_p
from parax.pipeline_stage import PipelineStage, XlaPipelineStage, slice_closed_jaxpr_by_pipeline_marks

# pylint: disable=redefined-builtin
unsafe_map, map = map, safe_map  # type: ignore


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
        devices: Mapping[str, Any],
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

