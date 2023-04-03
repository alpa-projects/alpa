"""Monkey patch other python libraries."""
# pylint: disable=protected-access, unused-argument
from functools import partial

import numpy as np
import jax
from jax import core, lax, numpy as jnp
from jax._src import dtypes, random as jax_src_random
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir.dialects import mhlo
from jax.core import Primitive
from jax.interpreters import pxla
from jax.interpreters import xla, mlir
from jax.lib import xla_bridge as jax_src_lib_xla_bridge
from jax.lib.xla_bridge import get_backend as default_get_backend
import flax

from alpa.global_env import global_config, is_worker

xops = xc._xla.ops

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
    jax_src_lib_xla_bridge.get_backend = override_get_backend
    jax.lib.xla_bridge.get_backend = override_get_backend

########################################
##### Monkey patch Jax
########################################


# Monkey patch random generator to use the stateful random generator.
# This can simplify the computational graph for dropout.
def fast_uniform(key, shape=(), dtype=dtypes.float_, minval=0.0, maxval=1.0):
    dtype = dtypes.canonicalize_dtype(dtype)
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
    return mhlo.RngOp(mu, sigma, shape,
                      mhlo.RngDistributionAttr.get("NORMAL")).results


mlir.register_lowering(rng_normal_p, _rng_normal_lowering)


def fast_normal(key, shape=(), dtype=dtypes.float_, mu=0.0, sigma=1.0):
    dtype = dtypes.canonicalize_dtype(dtype)
    shape = core.as_named_shape(shape)
    mu = jnp.asarray(mu, dtype)
    sigma = jnp.asarray(sigma, dtype)
    return rng_normal(mu, sigma, shape.positional)


def fast_truncated_normal(key, lower, upper, shape=None, dtype=dtypes.float_):
    dtype = dtypes.canonicalize_dtype(dtype)
    if shape is not None:
        shape = core.as_named_shape(shape)
    out = fast_normal(key, shape=shape, dtype=dtype)
    lower = lax.convert_element_type(lower, dtype)
    upper = lax.convert_element_type(upper, dtype)
    return jnp.clip(
        out,
        lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
        lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype)))


def fast_bernoulli(key, p=np.float32(0.5), shape=None):
    dtype = dtypes.canonicalize_dtype(lax.dtype(p))
    return jax.random.uniform(key, shape, dtype) < p


def remove_fold_in(key, data):
    return key


rng_primitives = [lax.rng_uniform_p, rng_normal_p]

# Monkey patch random generator to use the stateful random generator.
backup_random_uniform = jax.random.uniform
backup_random_truncated_normal = jax.random.truncated_normal
backup_random_normal = jax.random.normal
backup_random_bernoulli = jax.random.bernoulli
backup_random_foldin = jax.random.fold_in


def monkey_patch_random():
    jax.random.uniform = fast_uniform
    jax.random.truncated_normal = fast_truncated_normal
    jax.random.normal = fast_normal
    jax.random.bernoulli = fast_bernoulli
    jax.random.fold_in = remove_fold_in

    jax_src_random.uniform = fast_uniform
    jax_src_random.truncated_normal = fast_truncated_normal
    jax_src_random.normal = fast_normal
    jax_src_random.bernoulli = fast_bernoulli
    jax_src_random.fold_in = remove_fold_in


def restore_random():
    jax.random.uniform = backup_random_uniform
    jax.random.truncated_normal = backup_random_truncated_normal
    jax.random.normal = backup_random_normal
    jax.random.bernoulli = backup_random_bernoulli
    jax.random.fold_in = backup_random_foldin

    jax_src_random.uniform = backup_random_uniform
    jax_src_random.truncated_normal = backup_random_truncated_normal
    jax_src_random.normal = backup_random_normal
    jax_src_random.bernoulli = backup_random_bernoulli
    jax_src_random.fold_in = backup_random_foldin


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


pxla.ShardingSpec.__getstate__ = sharding_spec_getstate
pxla.ShardingSpec.__setstate__ = sharding_spec_setstate

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
    expanded = jax.nn.one_hot(inputs, self.embedding.shape[0], dtype=dtype)
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


flax.linen.Embed.setup = embed_setup
flax.linen.Embed.__call__ = embed_call_one_hot


# Monkey patch a new method "init_dummy" to flax's Module.
# This function initializes all weights with ones for testing/benchmark
# purposes.
# This function is much faster than the standard initialization.
def init_dummy(self, *args, **kwargs):
    avals = jax.eval_shape(self.init, *args, **kwargs)
    return jax.tree_util.tree_map(lambda x: jnp.full(x.shape, 1e-8, x.dtype),
                                  avals)


flax.linen.module.Module.init_dummy = init_dummy
