"""Monkey patch other python libraries."""
# pylint: disable=protected-access, unused-argument
from functools import partial

import numpy as np
import jax
from jax import core, lax, numpy as jnp
from jax._src import dtypes
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lib.xla_bridge import get_backend as default_get_backend
from jax.core import Primitive
from jax.interpreters import partial_eval as pe, pxla
from jax.interpreters import xla, mlir
from jax.interpreters.xla import (xops, jaxpr_subcomp, extend_name_stack,
                                  register_translation, wrap_name,
                                  _backend_specific_translations, parameter,
                                  xla_destructure, pyval_to_ir_constant)
import flax

from alpa.global_env import global_config, is_worker
from alpa.pipeline_parallel.primitive_def import xla_identity

########################################
##### Monkey patch the Jax backend
########################################

override_backend = None


def set_override_backend(backend):
    """Enable the JAX backend monkey patch."""
    global override_backend
    override_backend = backend


def override_get_backend(*args, **kwargs):
    """Override the `get_backend` in JAX to use PJRT backend managed by Alpa."""
    if override_backend is not None:
        return override_backend
    return default_get_backend(*args, **kwargs)


if is_worker:
    setattr(jax._src.lib.xla_bridge, "get_backend", override_get_backend)
    setattr(jax.lib.xla_bridge, "get_backend", override_get_backend)

########################################
##### Monkey patch Jax
########################################


# Monkey patch random generator to use the stateful random generator.
# This can simplify the computational graph for dropout.
def fast_uniform(key, shape=(), dtype=dtypes.float_, minval=0.0, maxval=1.0):
    shape = core.as_named_shape(shape)
    minval = jnp.asarray(minval, dtype)
    maxval = jnp.asarray(maxval, dtype)
    return lax.rng_uniform(minval, maxval, shape.positional)


def rng_normal(mu, sigma, shape):
    """Stateful PRNG generator. Experimental and its use is discouraged.

    Returns random numbers following normal distribution with (mu, sigma)

    You should use jax.random for most purposes; this function exists only for
    niche use cases with special performance requirements.

    This API may be removed at any time.
    """
    return rng_normal_p.bind(mu, sigma, shape=tuple(shape))


def _rng_normal_abstract_eval(mu, sigma, *, shape):
    if mu.dtype != sigma.dtype:
        raise ValueError(
            f"Arguments to rng_normal must have identical dtypes, got "
            f"{mu.dtype} and {sigma.dtype}.")
    if mu.shape != () or sigma.shape != ():
        raise ValueError(f"Arguments to rng_normal must be scalars; got shapes "
                         f"{mu.shape} and {sigma.shape}.")
    return mu.update(shape=shape,
                     dtype=mu.dtype,
                     weak_type=(mu.weak_type and sigma.weak_type))


def _rng_normal_translation_rule(ctx, avals_in, avals_out, mu, sigma, *, shape):
    c = ctx.builder
    xla_shape = xc.Shape.array_shape(c.get_shape(mu).xla_element_type(), shape)
    return [xops.RngNormal(mu, sigma, xla_shape)]


rng_normal_p = Primitive("rng_normal")
rng_normal_p.def_impl(partial(xla.apply_primitive, rng_normal_p))
rng_normal_p.def_abstract_eval(_rng_normal_abstract_eval)
xla.register_translation(rng_normal_p, _rng_normal_translation_rule)


def _rng_normal_lowering(ctx, mu, sigma, *, shape):
    aval_out, = ctx.avals_out
    shape, = mlir.ir_constants(np.array(aval_out.shape, np.int64),
                               canonicalize_types=False)
    return mhlo.RngNormalOp(mu, sigma, shape).results


mlir.register_lowering(rng_normal_p, _rng_normal_lowering)


def fast_normal(key, shape=(), dtype=dtypes.float_, mu=0.0, sigma=1.0):
    shape = core.as_named_shape(shape)
    mu = jnp.asarray(mu, dtype)
    sigma = jnp.asarray(sigma, dtype)
    return rng_normal(mu, sigma, shape.positional)


def fast_bernoulli(key, p=0.5, shape=None):
    dtype = dtypes.float_
    return jax.random.uniform(key, shape, dtype) < p


def remove_fold_in(key, data):
    return key


# Monkey patch random generator to use the stateful random generator.
jax._src.random.uniform = fast_uniform
jax.random.uniform = fast_uniform
jax._src.random.normal = fast_normal
jax.random.normal = fast_normal
jax._src.random.bernoulli = fast_bernoulli
jax.random.bernoulli = fast_bernoulli
jax._src.random.fold_in = remove_fold_in
jax.random.fold_in = remove_fold_in


# Monkey patch remat to use identity instead of while loop
def _zeros(c, xla_shape):
    if xla_shape.is_array():
        shape, dtype = xla_shape.dimensions(), xla_shape.numpy_dtype()
        zero = pyval_to_ir_constant(c, np.array(0, dtype=dtype))
        return xops.Broadcast(zero, shape)
    else:
        # It is a token
        return xops.CreateToken(c)


def _remat_using_while(ctx, in_nodes, name, call_jaxpr):
    """Lower remat to a single iteration while loop."""
    c = ctx.builder
    # Dummy subc for getting subcomp shapes.
    dummy_inputs = xops.Tuple(c, in_nodes)
    dummy_subc = xc.XlaBuilder("remat_dummy_subcomputation")
    dummy_input_op = parameter(dummy_subc,
                               0,
                               c.get_shape(dummy_inputs),
                               replicated=[])
    dummy_args = xla_destructure(dummy_subc, dummy_input_op)
    dummy_ctx = ctx.replace(builder=dummy_subc,
                            name_stack=extend_name_stack(
                                ctx.name_stack, wrap_name(name, "remat")))
    dummy_subcomp_outs = jaxpr_subcomp(dummy_ctx, call_jaxpr, (), *dummy_args)
    out_node_shapes = [dummy_subc.get_shape(o) for o in dummy_subcomp_outs]

    i_init = xops.Constant(c, np.array(0, dtype=np.int32))
    zeros_like_outs = [_zeros(c, s) for s in out_node_shapes]
    inputs = xops.Tuple(c, [i_init] + list(in_nodes) + zeros_like_outs)

    cond_subc = xc.XlaBuilder("remat_cond_subcomputation")
    input_op = parameter(cond_subc, 0, c.get_shape(inputs), replicated=[])
    i = xops.GetTupleElement(input_op, 0)
    rng = xops.RngUniform(xops.Constant(cond_subc, np.array(1, dtype=np.int32)),
                          xops.Constant(cond_subc, np.array(2, dtype=np.int32)),
                          xc.Shape.array_shape(xc.PrimitiveType.S32, []))
    cond_subc = cond_subc.build(xops.Lt(i, rng))

    body_subc = xc.XlaBuilder("remat_body_subcomputation")
    input_op = parameter(body_subc, 0, c.get_shape(inputs), replicated=[])
    i, *args = xla_destructure(body_subc, input_op)[:len(in_nodes) + 1]
    i_next = xops.Add(i, xops.Constant(body_subc, np.array(1, dtype=np.int32)))
    body_ctx = ctx.replace(builder=body_subc,
                           name_stack=extend_name_stack(
                               ctx.name_stack, wrap_name(name, "remat")))
    subcomp_outs = jaxpr_subcomp(body_ctx, call_jaxpr, (), *args)
    out_nodes = [i_next] + args + list(subcomp_outs)
    body_subc = body_subc.build(xops.Tuple(body_subc, out_nodes))
    outs = xops.While(cond_subc, body_subc, inputs)
    return xla_destructure(c, outs)[len(in_nodes) + 1:]


def _remat_using_identity(ctx, in_nodes, name, call_jaxpr):
    c = ctx.builder
    args = xla_identity(c, "remat_begin", *in_nodes)
    args = [xops.GetTupleElement(args, i) for i in range(len(in_nodes))]
    body_ctx = ctx.replace(
        name_stack=extend_name_stack(ctx.name_stack, wrap_name(name, "remat")))
    outs = jaxpr_subcomp(body_ctx, call_jaxpr, (), *args)
    # TODO: using an identity at the end can reduce little memory on 1 GPU,
    # but there are still some bugs
    # return xla_identity(c, op_type="remat_end", *outs)
    return outs


def _remat_translation_rule(ctx,
                            avals_in,
                            avals_out,
                            *in_nodes,
                            name,
                            call_jaxpr,
                            prevent_cse,
                            differentiated,
                            concrete,
                            policy,
                            device=None):
    del device, concrete, policy  # Unused.
    if differentiated and prevent_cse:
        if global_config.remat_using_while:
            return _remat_using_while(ctx, in_nodes, name, call_jaxpr)
        else:
            return _remat_using_identity(ctx, in_nodes, name, call_jaxpr)
    else:
        return jaxpr_subcomp(ctx, call_jaxpr, (), *in_nodes)


for dict_val in _backend_specific_translations.values():
    if pe.remat_call_p in dict_val:
        del dict_val[pe.remat_call_p]
register_translation(pe.remat_call_p, _remat_translation_rule)


# Support using pickle on ShardingSpec
def sharding_spec_getstate(self):
    sharding = []
    for x in self.sharding:
        if isinstance(x, pxla.NoSharding):
            sharding.append((0,))
        elif isinstance(x, pxla.Chunked):
            sharding.append((1, x.chunks))
        elif isinstance(x, pxla.Unstacked):
            sharding.append((2, x.size))
        else:
            raise ValueError(f"Invalid sharding: {x}")
    mesh_mapping = []
    for x in self.mesh_mapping:
        if isinstance(x, pxla.ShardedAxis):
            mesh_mapping.append((0, x.axis))
        elif isinstance(x, pxla.Replicated):
            mesh_mapping.append((1, x.replicas))
        else:
            raise ValueError(f"Invalid sharding: {x}")
    return (sharding, mesh_mapping)


def sharding_spec_setstate(self, state_tuple):
    sharding_encoding, mesh_mapping_encoding = state_tuple

    sharding = []
    for x in sharding_encoding:
        if x[0] == 0:
            sharding.append(pxla.NoSharding())
        elif x[0] == 1:
            sharding.append(pxla.Chunked(x[1]))
        elif x[0] == 2:
            sharding.append(pxla.Unstacked(x[1]))
        else:
            raise ValueError(f"Invalid sharding: {x}")

    mesh_mapping = []
    for x in mesh_mapping_encoding:
        if x[0] == 0:
            mesh_mapping.append(pxla.ShardedAxis(x[1]))
        elif x[0] == 1:
            mesh_mapping.append(pxla.Replicated(x[1]))
        else:
            raise ValueError(f"Invalid sharding: {x}")

    # pylint: disable=unnecessary-dunder-call
    self.__init__(
        sharding=sharding,
        mesh_mapping=mesh_mapping,
    )


setattr(pxla.ShardingSpec, "__getstate__", sharding_spec_getstate)
setattr(pxla.ShardingSpec, "__setstate__", sharding_spec_setstate)

# Monkey patch tree map to disable some warnings
jax._src.tree_util.tree_multimap = jax._src.tree_util.tree_map
jax.tree_multimap = jax._src.tree_util.tree_map

########################################
##### Monkey patch Flax
########################################


# Monkey patch the nn.Embed in flax to use onehot + matmul instead of
# gather/scatter,
# because we currently do not support 2d partition of gather/scatter.
def embed_call_one_hot(self, inputs):
    dtype = self.dtype
    if global_config.flax_always_use_fp16_embedding:
        dtype = jnp.float16
    expanded = jax.nn.one_hot(inputs, self.num_embeddings, dtype=dtype)
    ret = expanded @ jnp.asarray(self.embedding, dtype)
    return ret


# Monkey patch the nn.Embed in flax to add a fp16 conversion.
# This is used for manual pipeline marker.
def embed_setup(self):
    self.embedding = self.param("embedding", self.embedding_init,
                                (self.num_embeddings, self.features),
                                self.param_dtype)
    if self.dtype == jnp.float16:
        self.embedding_fp16 = self.embedding.astype(jnp.float16)


setattr(flax.linen.Embed, "setup", embed_setup)
setattr(flax.linen.Embed, "__call__", embed_call_one_hot)


# Monkey patch a new method "init_dummy" to flax's Module.
# This function initializes all weights with ones for testing/benchmark
# purposes.
# This function is much faster than the standard initialization.
def init_dummy(self, *args, **kwargs):
    avals = jax.eval_shape(self.init, *args, **kwargs)
    return jax.tree_util.tree_map(lambda x: jnp.full(x.shape, 1e-8, x.dtype),
                                  avals)


setattr(flax.linen.module.Module, "init_dummy", init_dummy)
