"""gshard based hybrid parallel"""
from collections import OrderedDict
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
from jax.core import ShapedArray
from jax.experimental.maps import mesh
from jax.experimental.pjit import pjit
from jax.interpreters import xla, partial_eval as pe
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
    print("jaxpr", jaxpr)
    exit(0)
    return compiled_func

