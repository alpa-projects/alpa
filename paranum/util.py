import jax
import numpy as np

def compute_bytes(pytree):
    flatten_args, _ = jax.tree_util.tree_flatten(pytree)
    ret = 0
    for x in flatten_args:
        if hasattr(x, "shape"):
            ret += np.prod(x.shape) * x.dtype.itemsize
    return ret

