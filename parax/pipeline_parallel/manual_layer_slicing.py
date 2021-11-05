"""Layer clustering and remat by layer."""
from functools import partial, wraps
from typing import List, Callable

from jax import tree_flatten, tree_unflatten
from jax._src.api import make_jaxpr
from jax.core import ClosedJaxpr, Jaxpr, jaxpr_as_fun, new_jaxpr_eqn
from jax.interpreters.partial_eval import remat_call_p

from parax.pipeline_parallel.computation import (
    slice_eqns_by_pipeline_marks, add_pipeline_marks_for_sliced_eqns)
from parax.pipeline_parallel.primitive_def import mark_pipeline_jaxpreqn
from parax.util import slices_to_jaxpr


def log_jaxpr(jaxpr, name):
    path = "/tmp/" + name
    with open(path, "w") as f:
        f.write(repr(jaxpr))


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


def manual_layer_slicing(fn: Callable, static_argnums=()):
    return transform_pipeline_forward(fn, add_pipeline_marks_for_sliced_eqns,
                                      static_argnums)


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


def remat(fn: Callable, static_argnums=(), use_pipeline=True):
    remat_fn = partial(remat_jaxpr, use_pipeline=use_pipeline)
    return transform_pipeline_forward(fn, remat_fn, static_argnums)


def insert_marker(origin_jaxpr, sliced_eqns):
    new_eqns = []
    for i, slices in enumerate(sliced_eqns):
        new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'start'))
        new_eqns.extend(slices)
        new_eqns.append(mark_pipeline_jaxpreqn([], [], str(i), 'end'))
    return ClosedJaxpr(
        Jaxpr(origin_jaxpr.jaxpr.constvars, origin_jaxpr.jaxpr.invars,
              origin_jaxpr.jaxpr.outvars, new_eqns), origin_jaxpr.consts)
