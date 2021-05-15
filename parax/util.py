"""Common utilities."""
import os

import flax
import numpy as np
from jax.api_util import shaped_abstractify
from jax.experimental.maps import FrozenDict
from jax.tree_util import tree_map, tree_flatten


########################################
##### API Utilities
########################################

def freeze_dict(pytree):
    """Convert a pytree to a FrozenDict."""
    def is_leaf(x):
        return isinstance(x, dict)

    def freeze(x):
        if isinstance(x, dict):
            return FrozenDict(x)
        return x

    return tree_map(freeze, pytree, is_leaf)


def auto_static_argnums(args):
    """Return the indices of static arguments according to heuristic rules."""
    def is_static_arg(arg):
        """Return whether an argument is a static argument according to heuristic rules."""
        if isinstance(arg, flax.optim.base.Optimizer):
            return False

        xs, _ = tree_flatten(arg)
        for x in xs:
            try:
                x = shaped_abstractify(x)
            except TypeError:
                return True
        return False

    return [i for i in range(len(args)) if is_static_arg(args[i])]


def auto_donate_argnums(args):
    """Return the indices of donated arguments according to heuristic rules."""
    def should_donate(x):
        # Always donate optimizer
        if isinstance(x, flax.optim.base.Optimizer):
            return True
        return False

    return [i for i in range(len(args)) if should_donate(args[i])]


########################################
##### Other Utilities
########################################

def run_cmd(cmd):
    """Run a bash commond."""
    print(cmd)
    os.system(cmd)


def compute_bytes(pytree):
    """Compute the total bytes of arrays in a pytree."""
    flatten_args, _ = tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret


########################################
##### Data Structure Utilities
########################################

def to_int_tuple(array):
    """Convert a numpy array to int tuple."""
    return tuple(int(x) for x in array)


def get_dim_last_value(array, dim):
    """Get the value of the last element in a dimension."""
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
