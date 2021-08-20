"""Monkey patch other python libraries."""

import flax
import jax
from jax import core, lax, numpy as jnp

from jax.lib.xla_bridge import get_backend as default_get_backend

########################################
##### Monkey patch the backend
########################################

override_backend = None


def override_get_backend(*args, **kwargs):
    """Override the `get_backend` in JAX to use PJRT backend managed by Parax."""
    global override_backend
    if override_backend is not None:
        return override_backend
    return default_get_backend(*args, **kwargs)


setattr(jax.lib.xla_bridge, "get_backend", override_get_backend)


def set_override_backend(backend):
    """Enable the JAX backend monkey patch."""
    global override_backend
    override_backend = backend


########################################
##### Monkey patch Jax
########################################


# Monkey patch random generator to use the stateful random generator.
# This can simplify the computational graph for dropout.
def fast_uniform(key, shape, dtype, minval=0.0, maxval=1.0):
    shape = core.as_named_shape(shape)
    minval = jnp.asarray(minval, dtype)
    maxval = jnp.asarray(maxval, dtype)
    return lax.rng_uniform(minval, maxval, shape.positional)


def remove_fold_in(key, data):
    return key


jax._src.random.uniform = fast_uniform
jax.random.uniform = fast_uniform
jax._src.random.fold_in = remove_fold_in
jax.random.fold_in = remove_fold_in


def _remat_using_bias(c, axis_env, in_nodes, name_stack, backend, name,
                      call_jaxpr):
    from jax.interpreters.xla import (xb, xc, xops, jaxpr_subcomp,
                                      extend_name_stack, wrap_name)
    import numpy as np
    rng_dict = {}

    def get_rng(c, node):
        dtype = c.get_shape(node).element_type()
        if dtype in rng_dict:
            return rng_dict[dtype]
        etype = c.get_shape(node).xla_element_type()
        rng = xops.RngUniform(xb.constant(c, np.array(1, dtype=dtype)),
                              xb.constant(c, np.array(1.001, dtype=dtype)),
                              xc.Shape.array_shape(etype, []))
        rng_dict[dtype] = rng
        return rng

    remat_subc = xb.make_computation_builder("remat_call_subcomputation")
    args = [
        xb.parameter(remat_subc, i, c.get_shape(n))
        for i, n in enumerate(in_nodes)
    ]
    bias_args = map(lambda node: xops.Add(node, get_rng(remat_subc, node)),
                    args)
    outs = jaxpr_subcomp(
        remat_subc, call_jaxpr, backend, axis_env, (),
        extend_name_stack(name_stack, wrap_name(name, "remat")), *bias_args)

    remat_subc = remat_subc.build(xops.Tuple(remat_subc, outs))
    return xops.Call(c, remat_subc, list(in_nodes))


jax.xla._remat_using_while = _remat_using_bias

########################################
##### Monkey patch Flax
########################################


# Monkey patch the nn.Embed in flax to use onehot + matmul instead of gather/scatter.
# Because we currently do not support 2d partition of gather/scatter.
def embed_call_one_hot(self, inputs):
    expanded = jax.nn.one_hot(inputs, self.num_embeddings, dtype=self.dtype)
    ret = expanded @ jnp.asarray(self.embedding, self.dtype)
    return ret


# Monkey patch the nn.Embed in flax to use always use fp32 as parameter type
def embed_setup(self):
    self.embedding = self.param('embedding', self.embedding_init,
                                (self.num_embeddings, self.features))


setattr(flax.linen.Embed, "setup", embed_setup)
setattr(flax.linen.Embed, "__call__", embed_call_one_hot)

# Monkey patch to make the nn.LayerNorm an identity function.
# This is used for debugging
#def layer_norm_identity(self, inputs):
#    return inputs
#
#
#setattr(flax.linen.LayerNorm, "__call__", layer_norm_identity)


# Mondey patch a new method "init_dummy" to flax's Module.
# This function initializes all weights with ones for testing/benchmark purposes.
# This function is much faster than the standard initialization.
def init_dummy(self, *args, **kwargs):
    avals = jax.eval_shape(self.init, *args, **kwargs)
    return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype), avals)


setattr(flax.linen.module.Module, "init_dummy", init_dummy)
