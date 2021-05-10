import os

import flax
import jax
from jax.api_util import shaped_abstractify
from jax.tree_util import tree_map, tree_flatten
from jax.experimental.maps import FrozenDict
import numpy as np


def compute_bytes(pytree):
    """Compute the total bytes of arrays in a pytree"""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


def freeze_dict(pytree):
    """Convert a pytree to a FrozenDict"""
    def is_leaf(x):
        return isinstance(x, dict)

    def freeze(x):
        if isinstance(x, dict):
            return FrozenDict(x)

    return tree_map(freeze, pytree, is_leaf)


def auto_static_argnums(args):
    """Return the indices of static arguments"""
    def is_static_arg(x):
        """Return whether an argument is a static argument according to heuristic rules"""
        if isinstance(x, flax.optim.base.Optimizer):
            return False

        xs, _ = tree_flatten(x)
        for x in xs:
            try:
                x = shaped_abstractify(x)
            except TypeError:
                return True
        return False

    return [i for i in range(len(args)) if is_static_arg(args[i])]


def auto_donate_argnums(args):
    """Return the indices of donated arguments"""
    def should_donate(x):
        # Always donate optimizer
        if isinstance(x, flax.optim.base.Optimizer):
            return True

    return [i for i in range(len(args)) if should_donate(args[i])]


def run_cmd(cmd):
    """Run a bash commond"""
    print(cmd)
    os.system(cmd)


def get_dim_last_value(array, dim):
    """Get the value of the last element in a dimension"""
    indices = tuple(0 if i != dim else array.shape[dim] - 1 for i in range(len(array.shape)))
    return array[indices]


class FastLookupList:
    def __init__(self, iterable=()):
        self.elements = list(iterable)
        self.elements_set = set(iterable)

    def __getitem__(self, key):
        return self.elements[key]

    def __len__(self):
        return len(self.elements)

    def __contains__(self, element):
        return element in self.elements_set

    def append(self, element):
        self.elements.append(element)
        self.elements_set.add(element)
