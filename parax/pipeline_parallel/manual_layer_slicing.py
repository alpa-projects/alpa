"""Layer clustering and remat by layer."""
from functools import partial, wraps
from typing import Callable

from jax import tree_flatten, tree_unflatten
from jax._src.api import make_jaxpr
from jax.core import Var, Jaxpr, ClosedJaxpr, DropVar, Literal, jaxpr_as_fun, new_jaxpr_eqn, gensym
from jax.interpreters.partial_eval import remat_call_p

from parax.pipeline_parallel.primitive_def import (pipeline_p,
                                                   mark_pipeline_jaxpreqn)
from parax.util import slices_to_jaxpr


def get_var_mapping(mapping, var):
    if isinstance(var, Var) and var in mapping:
        return mapping[var]
    else:
        return var


def slice_eqns_by_pipeline_marks(closed_jaxpr: ClosedJaxpr):
    sliced_eqns = []
    current_computation_eqns = None

    for eqn in closed_jaxpr.jaxpr.eqns:
        if eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'start':
            assert current_computation_eqns is None, "Defining a pipeline computation inside a pipeline computation is not allowed."
            current_computation_eqns = []
        elif eqn.primitive is pipeline_p and eqn.params['mark_type'] == 'end':
            assert current_computation_eqns is not None, "Ending a pipeline computation before its start."
            sliced_eqns.append(current_computation_eqns)
            current_computation_eqns = None
        else:
            assert current_computation_eqns is not None
            current_computation_eqns.append(eqn)
    assert current_computation_eqns is None
    return sliced_eqns


def transform_pipeline_forward(fn: Callable, transform_fn, static_argnums=()):

    @wraps(fn)
    def wrapped(*args):
        origin_jaxpr, out_shape_tree = make_jaxpr(fn,
                                                  static_argnums=static_argnums,
                                                  return_shape=True)(*args)
        sliced_eqns = slice_eqns_by_pipeline_marks(origin_jaxpr)
        flatten_args, _ = tree_flatten(args)
        new_jaxpr = transform_fn(origin_jaxpr, sliced_eqns)
        ans = jaxpr_as_fun(new_jaxpr)(*flatten_args)
        _, out_tree = tree_flatten(out_shape_tree)
        return tree_unflatten(out_tree, ans)

    return wrapped


def add_pipeline_marks_for_sliced_eqns(closed_jaxpr: ClosedJaxpr, sliced_eqns):
    n_layers = len(sliced_eqns)
    layer_pipeline_invars = [set() for _ in range(n_layers)]
    layer_pipeline_outvars = [set() for _ in range(n_layers)]
    var_layer_dict = {}

    for var in closed_jaxpr.jaxpr.invars:
        var_layer_dict[var] = -1

    for i, eqns in enumerate(sliced_eqns):
        for eqn in eqns:
            for var in eqn.invars:
                if (not isinstance(var, Literal) and
                        var not in closed_jaxpr.jaxpr.constvars and
                        var_layer_dict[var] != i):
                    layer_pipeline_invars[i].add(var)
                    if var_layer_dict[var] == -1:
                        var_layer_dict[var] = i
                    layer_pipeline_outvars[var_layer_dict[var]].add(var)
            for var in eqn.outvars:
                if not isinstance(var, DropVar):
                    var_layer_dict[var] = i

    for var in closed_jaxpr.jaxpr.outvars:
        if (not isinstance(var, Literal) and
                var not in closed_jaxpr.jaxpr.constvars and
                var_layer_dict[var] != -1):
            layer_pipeline_outvars[var_layer_dict[var]].add(var)

    gensym_func = gensym([closed_jaxpr.jaxpr])
    var_mapping = {}

    new_eqns = []
    for i, eqns in enumerate(sliced_eqns):
        # pipeline start eqn
        computation_var_mapping = {}

        pipeline_start_invars = []
        pipeline_start_outvars = []
        for var in layer_pipeline_invars[i]:
            new_var = gensym_func(var.aval)
            pipeline_start_invars.append(get_var_mapping(var_mapping, var))
            pipeline_start_outvars.append(new_var)
            computation_var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_start_invars,
                                   pipeline_start_outvars, str(i), 'start'))
        # all other eqns
        for eqn in eqns:
            new_invars = [
                get_var_mapping(computation_var_mapping, var)
                for var in eqn.invars
            ]
            new_eqns.append(
                new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive,
                              eqn.params, eqn.source_info))
        # pipeline end eqn
        pipeline_end_invars = []
        pipeline_end_outvars = []
        for var in layer_pipeline_outvars[i]:
            new_var = gensym_func(var.aval)
            pipeline_end_invars.append(
                get_var_mapping(computation_var_mapping, var))
            pipeline_end_outvars.append(new_var)
            var_mapping[var] = new_var
        new_eqns.append(
            mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars,
                                   str(i), 'end'))
    new_jaxpr = Jaxpr(
        closed_jaxpr.jaxpr.constvars,
        closed_jaxpr.jaxpr.invars,
        [
            get_var_mapping(var_mapping, var)
            for var in closed_jaxpr.jaxpr.outvars
        ],
        new_eqns,
    )
    new_closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
    return new_closed_jaxpr


def remat_jaxpr(origin_jaxpr, sliced_eqns, use_pipeline):
    """
    The input function should be marked by pipeline markers without input
    """
    sliced_jaxprs = slices_to_jaxpr(origin_jaxpr, sliced_eqns)
    new_eqns = []
    for i, jaxpr in enumerate(sliced_jaxprs):
        if use_pipeline:
            new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'start'))
        new_invars = jaxpr.jaxpr.invars + jaxpr.jaxpr.constvars
        new_jaxpr = Jaxpr([], new_invars, jaxpr.jaxpr.outvars, jaxpr.jaxpr.eqns)
        new_eqns.append(
            new_jaxpr_eqn(
                new_invars, new_jaxpr.outvars, remat_call_p,
                dict(concrete=False,
                     differentiated=False,
                     name=str(i),
                     call_jaxpr=new_jaxpr,
                     prevent_cse=True,
                     policy=None)))
        if use_pipeline:
            new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'end'))
    new_closed_jaxpr = ClosedJaxpr(
        Jaxpr(origin_jaxpr.jaxpr.constvars, origin_jaxpr.jaxpr.invars,
              origin_jaxpr.jaxpr.outvars, new_eqns), origin_jaxpr.consts)
    return new_closed_jaxpr


def insert_marker(origin_jaxpr, sliced_eqns):
    new_eqns = []
    for i, slices in enumerate(sliced_eqns):
        new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'start'))
        new_eqns.extend(slices)
        new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'end'))
    return ClosedJaxpr(
        Jaxpr(origin_jaxpr.jaxpr.constvars, origin_jaxpr.jaxpr.invars,
              origin_jaxpr.jaxpr.outvars, new_eqns), origin_jaxpr.consts)


def manual_layer_slicing(fn: Callable, static_argnums=()):
    """TODO(yonghao): always document top-level API."""
    return transform_pipeline_forward(fn, add_pipeline_marks_for_sliced_eqns,
                                      static_argnums)


def remat(fn: Callable, static_argnums=(), use_pipeline=True):
    """TODO(yonghao): always document top-level API."""
    remat_fn = partial(remat_jaxpr, use_pipeline=use_pipeline)
    return transform_pipeline_forward(fn, remat_fn, static_argnums)
