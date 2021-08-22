"""pipeline parallel on a single device."""
from typing import Sequence, Set, Mapping, Any, Dict

import jax
from jax import linear_util as lu
from jax._src.util import safe_map
from jax.core import Var, DropVar, ClosedJaxpr, Literal, gensym
from jax.interpreters import partial_eval as pe

from parax.pipeline_parallel.primitive_def import pipeline_p
from parax.pipeline_parallel.stage import PipelineStage, XlaPipelineStage, slice_closed_jaxpr_by_manual_pipeline_marks

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

    def run_stage(self,
                  stage: PipelineStage,
                  invals: Dict[Var, Any]):
        """
        Run a pipeline stage.

        Args:
            stage (PipelineStage): The pipeline stage to run.
            invals (Set[Var], optional): Input value dict.

        Returns:
            Two dictionaries with values of pipeline & global output variables.
        """
        runnable = stage.get_runnable()
        invals_list = []
        for var in stage.invars:
            invals_list.append(invals[var])
        outvals_list = runnable(*invals_list)
        outvals = dict(zip(stage.outvars, outvals_list))
        self.env.update(outvals)

    def get_val(self, var):
        return self.env[var]


def local_pipeline_runtime(pipeline_stages: Sequence[PipelineStage],
                           global_invars: Sequence[Var],
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
        runners = {}
        # Mark pipeline variables
        var_stage_mapping = {}
        for stage in pipeline_stages:
            stage_invals = {}
            for var in stage.invars:
                if var in global_invals:
                    stage_invals[var] = global_invals[var]
                else:
                    assert var in var_stage_mapping, f"referred to an unknown var {var}"
                    stage_invals[var] = runners[var_stage_mapping[var]].get_val(var)
            if stage.name not in runners:
                runners[stage.name] = LocalPipelineRunner(
                    stage.name, global_invals)
            runners[stage.name].run_stage(stage, stage_invals)
            for var in stage.outvars:
                var_stage_mapping[var] = stage.name

        global_outvals_list = []
        for var in global_outvars:
            if isinstance(var, Literal):
                global_outvals_list.append(var.val)
            else:
                assert var in var_stage_mapping, f"referred to an unknown var {var}"
                global_outvals_list.append(runners[var_stage_mapping[var]].get_val(var))
        return global_outvals_list

    return ret_func


@lu.cache
def local_pipeline_parallel_callable(fun: lu.WrappedFun,
                                     devices: Mapping[str, Any], *avals):
    """Pipeline parallel callable."""
    with jax.disable_jit():
        jaxpr, _, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    jax_pipeline_stages = slice_closed_jaxpr_by_manual_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    xla_pipeline_stages = [
        XlaPipelineStage.from_jax_pipeline_stage(stage)
        for stage in jax_pipeline_stages
    ]
    return local_pipeline_runtime(xla_pipeline_stages, global_invars,
                                  global_outvars)
