"""Monkey patch other python libraries."""

import jax
from jax import core, lax

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


# DEPRECATED!
# Patch __eq__ and __hash__ function for OptimizerDef in flax

#def __hash__(self):
#    return hash(self.hyper_params.learning_rate)
#def __eq__(self, other):
#    return True
#setattr(flax.optim.GradientDescent, "__hash__", __hash__)
#setattr(flax.optim.GradientDescent, "__eq__", __eq__)
