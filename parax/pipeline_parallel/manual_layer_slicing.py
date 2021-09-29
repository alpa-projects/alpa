"""Layer clustering and remat by layer."""
from functools import wraps
from typing import List, Callable

from jax import tree_flatten, tree_unflatten
from jax._src.api import make_jaxpr
from jax.core import jaxpr_as_fun

from parax.pipeline_parallel.stage import (slice_eqns_by_pipeline_marks,
                                           add_pipeline_marks_for_sliced_eqns)


def log_jaxpr(jaxpr, name):
    path = "/tmp/" + name
    with open(path, "w") as f:
        f.write(repr(jaxpr))


def manual_layer_slicing(fn: Callable, static_argnums=()):

    @wraps(fn)
    def wrapped(*args):
        origin_jaxpr, out_shape_tree = make_jaxpr(fn,
                                                  static_argnums=static_argnums,
                                                  return_shape=True)(*args)
        sliced_eqns = slice_eqns_by_pipeline_marks(origin_jaxpr)
        new_jaxpr = add_pipeline_marks_for_sliced_eqns(origin_jaxpr,
                                                       sliced_eqns)
        flatten_args, _ = tree_flatten(args)
        ans = jaxpr_as_fun(new_jaxpr)(*flatten_args)
        _, out_tree = tree_flatten(out_shape_tree)
        return tree_unflatten(out_tree, ans)

    return wrapped
