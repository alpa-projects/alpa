"""Monkey patch other python libraries."""

import flax
import jax
from jax import core, lax, numpy as jnp


# Monkey patch random generator to use the stateful random generator.
# This can simplify the computational graph for dropout.
def fast_uniform(key, shape, dtype, minval=0.0, maxval=1.0):
    shape = core.as_named_shape(shape)
    return lax.rng_uniform(minval, maxval, shape.positional)


def remove_fold_in(key, data):
    return key


jax._src.random.uniform = fast_uniform
jax.random.uniform = fast_uniform
jax._src.random.fold_in = remove_fold_in
jax.random.fold_in = remove_fold_in


# Mondey patch a new method "init_dummy" to flax's Module.
# This function initializes all weights with ones for testing/benchmark purposes.
# This function is much faster than the standard initialization.
def init_dummy(self, *args, **kwargs):
    avals = jax.eval_shape(self.init, *args, **kwargs)
    return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype), avals)


setattr(flax.linen.module.Module, "init_dummy", init_dummy)
