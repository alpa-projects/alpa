"""gshard based hybrid parallel"""
import cloudpickle

import jax
from jax import jit
from jax import linear_util as lu
from jax.core import Atom, Var, DropVar, JaxprEqn, Jaxpr, ClosedJaxpr, Primitive, Literal, abstract_unit, jaxpr_as_fun
from jax.interpreters import xla, ad, partial_eval as pe
from jax.lib import xla_client as xc
from jax._src.util import safe_map

unsafe_map, map = map, safe_map  # type: ignore

from parax.pipeline_primitive_def import *

# Note: import after the above lines
from parax.pipe import JaxPipeline
from parax.pipeline_stage import PipelineStage, PicklableStage

def slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr):
    # We assume the closed_jaxpr includes pipeline start and end markers. Also,
    # the variables in the markers represents the variables being sent
    # through the network. While other input variables must be directly from
    # the invars.
    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_outvars = set(var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
    global_consts_dir = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    var2stage = {}
    result_stages = []

    current_stage = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_stage is None, "Defining a pipeline stage inside a pipeline stage is not allowed."
            current_stage = PipelineStage(name=eqn.params['name'])
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
    return result_stages

class LocalPipelineRunner:
    def __init__(self, name, global_invals):
        self.name = name
        self.env = {}
        self.global_invals = global_invals

    def run_stage(self, stage, prev_stage_pipeline_outvals=None):
        closed_jaxpr = stage.closed_jaxpr()
        runnable = jit(jaxpr_as_fun(closed_jaxpr))
        invals_list = []
        for var in closed_jaxpr.jaxpr.invars:
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
        outvals = {var: val for var, val in zip(closed_jaxpr.jaxpr.outvars, outvals_list)}
        local_outvals = {var: outvals[var] for var in stage.local_outvars}
        pipeline_outvals = {var: outvals[var] for var in stage.pipeline_outvars}
        global_outvals = {var: outvals[var] for var in stage.global_outvars}
        self.env.update(local_outvals)
        return pipeline_outvals, global_outvals

def local_pipeline_runtime(pipeline_stages, global_invars, global_outvars):
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
    pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    # pickable_pipeline_stages = []
    # for stage in pipeline_stages:
    #     # pickable_stage = PicklableStage.from_pipeline_stage(stage)
    #     # pickle and unpickle
    #     # picked_stage = cloudpickle.dumps(pickable_stage)
    #     # new_pickable_stage = cloudpickle.loads(picked_stage)
    #     pickable_pipeline_stages.append(pickable_stage)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    return local_pipeline_runtime(pipeline_stages, global_invars, global_outvars)

@lu.cache
def distributed_pipeline_parallel_callable(
    fun: lu.WrappedFun,
    *avals
):

    with jax.disable_jit():
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    jp = JaxPipeline(pipeline_stages=pipeline_stages,
                     global_invars=global_invars,
                     global_outvars=global_outvars)
    return lambda *args, **kwargs: jp.run(*args, **kwargs)
