import os

import flax
import jax
from jax.api_util import shaped_abstractify
from jax.tree_util import tree_map, tree_flatten
from jax.experimental.maps import FrozenDict
import numpy as np


def compute_bytes(pytree):
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def freeze_dict(pytree):
    def is_leaf(x):
        return isinstance(x, dict)

    def freeze(x):
        if isinstance(x, dict):
            return FrozenDict(x)

    return tree_map(freeze, pytree, is_leaf)


def is_static_arg(x):
    if isinstance(x, flax.optim.base.Optimizer):
        return False

    xs, _ = tree_flatten(x)
    for x in xs:
        try:
            x = shaped_abstractify(x)
        except TypeError:
            return True
    return False


def auto_static_argnums(args):
    """Return the indices of static arguments"""
    return [i for i in range(len(args)) if is_static_arg(args[i])]


def run_cmd(cmd):
    print(cmd)
    os.system(cmd)

