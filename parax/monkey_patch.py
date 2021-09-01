"""Monkey patch other python libraries."""

import flax
from flax.linen.module import compact, wrap_method_once
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


def _remat_using_identity(c, axis_env, in_nodes, name_stack, backend, name,
                          call_jaxpr):
    from jax.interpreters.xla import (xops, jaxpr_subcomp, extend_name_stack,
                                      wrap_name)
    from jax.lib import xla_client as xc

    def all_index(shape, cur):
        out = []
        if shape.is_tuple():
            for i, subshape in enumerate(shape.tuple_shapes()):
                out.extend(all_index(subshape, cur + [i]))
        elif shape.is_array():
            out.append(xc.ShapeIndex(cur))
        return out

    def id(c, *args):
        input_params = xc.ops.Tuple(c, args)
        input_shape = c.get_shape(input_params)
        aliasing = [(index, (0, index)) for index in all_index(input_shape, [])]
        output_tuple = xc.ops.CustomCallWithOnlyAliasing(
            c,
            b'identity',
            operands=(input_params,),
            shape=input_shape,
            output_operand_aliasing=aliasing)
        return output_tuple

    bias_args = id(c, *in_nodes)
    bias_args = [
        xops.GetTupleElement(bias_args, i) for i in range(len(in_nodes))
    ]
    outs = jaxpr_subcomp(
        c, call_jaxpr, backend, axis_env, (),
        extend_name_stack(name_stack, wrap_name(name, "remat")), *bias_args)

    return (xops.Tuple(c, outs))


jax.xla._remat_using_while = _remat_using_identity

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


# Mondey patch nn.LayerNorm in flax to make sure all gradients are in fp16
# when using mixed-precision.
@compact
def layer_norm_call(self, x):
    x = jnp.asarray(x, jnp.float32)
    features = x.shape[-1]
    mean = jnp.mean(x, axis=-1, keepdims=True)
    mean2 = jnp.mean(lax.square(x), axis=-1, keepdims=True)
    var = mean2 - lax.square(mean)
    mul = lax.rsqrt(var + self.epsilon)
    mul = jnp.asarray(mul, self.dtype)
    if self.use_scale:
        mul = mul * jnp.asarray(
            self.param('scale', self.scale_init, (features,)), self.dtype)
    y = (x - mean) * mul
    y = jnp.asarray(y, self.dtype)
    if self.use_bias:
        y = y + jnp.asarray(self.param('bias', self.bias_init,
                                       (features,)), self.dtype)
    return jnp.asarray(y, self.dtype)


setattr(flax.linen.LayerNorm, "__call__", wrap_method_once(layer_norm_call))


# Mondey patch a new method "init_dummy" to flax's Module.
# This function initializes all weights with ones for testing/benchmark purposes.
# This function is much faster than the standard initialization.
def init_dummy(self, *args, **kwargs):
    avals = jax.eval_shape(self.init, *args, **kwargs)
    return jax.tree_util.tree_map(lambda x: jnp.ones(x.shape, x.dtype), avals)


setattr(flax.linen.module.Module, "init_dummy", init_dummy)
