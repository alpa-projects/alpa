from jax._src.tree_util import tree_flatten

import numpy as np

from jax import make_jaxpr
from jax.core import jaxpr_as_fun
import jax.numpy as jnp
from flax import optim

from parax.pipeline_parallel.stage import slice_apply_gradient_by_invars

M, N, P, Q = 10, 15, 20, 30
# An MLP
params = {
  'f1' : np.random.rand(M, N),
  'b1' : np.random.rand(N),
  'f2' : np.random.rand(N, P),
  'b2' : np.random.rand(P),
  'f3' : np.random.rand(P, Q),
  'b3' : np.random.rand(Q)
  }
grad = {
  'f1' : np.random.rand(M, N),
  'b1' : np.random.rand(N),
  'f2' : np.random.rand(N, P),
  'b2' : np.random.rand(P),
  'f3' : np.random.rand(P, Q),
  'b3' : np.random.rand(Q)
}
# define the jaxpr for apply_gradient phase
optimizer = optim.Adam(1e-2).create(params)
def apply_grad(optimizer, grad):
  new_optimizer = optimizer.apply_gradient(grad)
  return new_optimizer
closed_jaxpr = make_jaxpr(apply_grad)(optimizer, grad)
# Assume that each layer is on a mesh
grad_mesh = dict()
invars = closed_jaxpr.jaxpr.invars
for i in range(1, 4):
  grad_mesh[invars[-1 * i]] = 3 - i
  grad_mesh[invars[-1 * i - 3]] = 3 - i
# get slices and infered allocation for all invars
slices, infer = slice_apply_gradient_by_invars(closed_jaxpr, grad_mesh)
# if print infer here, only the first invar(an int scalar) is replicated in multiple meshes.
# get the actual answer
ans, _ = tree_flatten(apply_grad(optimizer, grad))
# compute per slices
inputs = {var : aval for var, aval in zip(invars, tree_flatten((optimizer, grad))[0])}
outputs = dict()
for slice in slices:
    args = []
    for arg in slice.jaxpr.invars:
        args.append(inputs[arg])
    outs = jaxpr_as_fun(slice)(*args)
    for var, aval in zip(slice.jaxpr.outvars, outs):
        if var not in outputs:
            outputs[var] = aval
        else:
            # may compute at multiple slices, check they get the same result
            assert jnp.allclose(outputs[var], aval)
# check the correctness of the result
flatten_outs = [outputs[var] for var in closed_jaxpr.jaxpr.outvars]
for cor, tes in zip(ans, flatten_outs):
    assert jnp.allclose(cor, tes)