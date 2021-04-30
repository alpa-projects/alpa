"""gshard based hybrid parallel"""
from dataclasses import dataclass, field
from typing import List, Set, Any, Dict

import numpy as np

import jax
from jax import jit
from jax import linear_util as lu
from jax.core import Atom, Var, DropVar, JaxprEqn, Jaxpr, ClosedJaxpr, Primitive, Literal, abstract_unit, jaxpr_as_fun
from jax.interpreters import xla, ad, partial_eval as pe
from jax.lib import xla_client as xc
from jax._src.util import safe_map

unsafe_map, map = map, safe_map  # type: ignore

pipeline_p = Primitive('pipeline')
pipeline_p.multiple_results = True

def mark_pipeline(*args, name, mark_type):
    if mark_type not in ('start', 'end', 'jvp_start', 'jvp_end'):
        raise ValueError('Unknown mark type: %s' % mark_type)
    return pipeline_p.bind(*args, name=name, mark_type=mark_type)

def _pipeline_impl(*args, **kwargs):
    # The pipeline marker acts as an identity function
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
class PipelineStage:
    name: str
    eqns: List[JaxprEqn] = field(default_factory=list)
    consts_dir: Dict[Atom, Any] = field(default_factory=dict)
    # invars
    pipeline_invars: Set[Var] = field(default_factory=set)
    global_invars: Set[Var] = field(default_factory=set)
    local_invars: Set[Var] = field(default_factory=set)
    # outvars
    pipeline_outvars: Set[Var] = field(default_factory=set)
    global_outvars: Set[Var] = field(default_factory=set)
    local_outvars: Set[Var] = field(default_factory=set)
    # intermediate vars
    intermediate_vars: Set[Var] = field(default_factory=set)

    def closed_jaxpr(self):
        jaxpr = Jaxpr(
            constvars=self.consts_dir.keys(),
            invars=list(self.pipeline_invars | self.global_invars | self.local_invars),
            outvars=list(self.pipeline_outvars | self.global_outvars | self.local_outvars),
            eqns=self.eqns,
        )
        closed_jaxpr = ClosedJaxpr(jaxpr, self.consts_dir.values())
        return closed_jaxpr


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
    global_invars = closed_jaxpr.jaxpr.invars
    global_outvars = closed_jaxpr.jaxpr.outvars
    return local_pipeline_runtime(pipeline_stages, global_invars, global_outvars)
