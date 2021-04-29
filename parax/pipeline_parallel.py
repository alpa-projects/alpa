"""gshard based hybrid parallel"""
from collections import OrderedDict, namedtuple
from functools import wraps, partial
import itertools
import os
import re
import threading
from dataclasses import dataclass, field
from typing import List, Set, Any, Dict

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
from jax.core import Atom, Var, DropVar, JaxprEqn, Jaxpr, ClosedJaxpr, Primitive, Literal, ShapedArray, abstract_unit
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
from parax.util import FastLookupList
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
    print("=" * 30 + " closed_jaxpr " + "=" * 30)
    print(closed_jaxpr)
    pipeline_stages = slice_closed_jaxpr_by_pipeline_marks(closed_jaxpr)
    for stage in pipeline_stages:
        print("=" * 30 + " stage " + stage.name + " " + "=" * 30)
        print("consts_dir", stage.consts_dir)
        print("pipeline_invars", stage.pipeline_invars)
        print("global_invars", stage.global_invars)
        print("local_invars", stage.local_invars)
        print("pipeline_outvars", stage.pipeline_outvars)
        print("global_outvars", stage.global_outvars)
        print("local_outvars", stage.local_outvars)
        print("intermediate_vars", stage.intermediate_vars)
        print(stage.closed_jaxpr())
    exit(0)
    return compiled_func

