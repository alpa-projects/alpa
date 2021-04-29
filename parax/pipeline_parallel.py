"""gshard based hybrid parallel"""
from collections import OrderedDict, namedtuple
from functools import wraps, partial
import itertools
import os
import re
import threading

import numpy as np

import jax
from jax import linear_util as lu
from jax.api_util import (
    shaped_abstractify,
    flatten_fun,
    flatten_axes,
    flatten_fun_nokwargs,
    argnums_partial,
)
from jax.config import flags, config, bool_env
from jax.core import Jaxpr, ClosedJaxpr, Primitive, Literal, ShapedArray, abstract_unit
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit
from jax.interpreters import xla, ad, partial_eval as pe
from jax.interpreters.pxla import parallel_callable, mesh_callable, Mesh
from jax.interpreters.sharded_jit import PartitionSpec
from jax.lib import xla_bridge as xb, xla_client as xc
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
from jax._src.util import (
    unzip2,
    curry,
    partial,
    safe_map,
    safe_zip,
    prod,
    split_list,
    extend_name_stack,
    wrap_name,
    cache,
    wraps,
    HashableFunction,
)

from parax import util, global_config
from parax.auto_sharding import auto_sharding_callable
from parax.pmap_data_parallel import should_replicate_map, should_replicate_is_leaf

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

PipelineStage = namedtuple("PipelineStage", ["name", "closed_jaxpr", "pipeline_invars", "local_invars", "pipeline_outvars", "local_outvars"])

def slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr):
    # We assume the closed_jaxpr includes pipeline start and end markers. Also,
    # the variables in the markers represents the variables being sent
    # through the network. While other input variables must be directly from
    # the invars.
    global_invars = set(closed_jaxpr.jaxpr.invars)
    global_outvars = set(closed_jaxpr.jaxpr.outvars)
    global_consts_dir = OrderedDict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
    result_pipeline_stages = []

    in_pipeline_stage = False

    for index, eqn in enumerate(closed_jaxpr.jaxpr.eqns):
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert not in_pipeline_stage, "Defining a pipeline stage inside a pipeline stage is not allowed."
            in_pipeline_stage = True
            stage_name = eqn.params['name']
            stage_consts_dir = OrderedDict()
            stage_pipeline_invars = list(eqn.invars)
            stage_local_invars = []
            stage_local_outvars = []
            stage_eqns = []
            stage_intermediate_vars = set()

        assert in_pipeline_stage

        for var in eqn.invars:
            if isinstance(var, Literal):
                continue
            elif var in global_consts_dir:
                if var not in stage_consts_dir:
                    stage_consts_dir[var] = global_consts_dir[var]
            elif (var in stage_pipeline_invars) or (var in stage_intermediate_vars):
                continue
            elif var in global_invars:
                if var not in stage_local_invars:
                    stage_local_invars.append(var)
            else:
                raise ValueError("Variable {} is not indicated as a pipeline stage input.".format(var))

        stage_intermediate_vars.update(eqn.outvars)
        for var in eqn.outvars:
            if var in global_outvars:
                stage_local_outvars.append(var)

        stage_eqns.append(eqn)

        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert in_pipeline_stage, "Ending a pipeline stage before its start."
            assert stage_name == eqn.params['name'], "Ending a pipeline stage different from its start."
            in_pipeline_stage = False
            stage_pipeline_outvars = list(eqn.outvars)
            stage_jaxpr = Jaxpr(
                constvars=stage_consts_dir.keys(),
                invars=stage_pipeline_invars + stage_local_invars,
                outvars=stage_pipeline_outvars + stage_local_outvars,
                eqns=stage_eqns,
            )
            stage_closed_jaxpr = ClosedJaxpr(stage_jaxpr, stage_consts_dir.values())
            current_pipeline_stage = PipelineStage(
                name=stage_name,
                closed_jaxpr=stage_closed_jaxpr,
                pipeline_invars=stage_pipeline_invars,
                local_invars=stage_local_invars,
                pipeline_outvars=stage_pipeline_outvars,
                local_outvars=stage_local_outvars,
            )
            result_pipeline_stages.append(current_pipeline_stage)
    return result_pipeline_stages

@lu.cache
def pipeline_parallel_callable(
    fun: lu.WrappedFun,
    in_tree,
    out_tree_thunk,
    devices,
    donated_invars,
    memory_budget_per_device,
    *avals
):
    with jax.disable_jit():
        jaxpr, out_avals, consts = pe.trace_to_jaxpr_final(fun, avals)
    closed_jaxpr = ClosedJaxpr(jaxpr, consts)
    print("closed_jaxpr", closed_jaxpr)
    pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    print(pipeline_stages)
    exit(0)
    return compiled_func

