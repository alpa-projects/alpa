from alpa.pipeline_parallel.apply_grad import ApplyGradRewriter, slice_apply_gradient
from alpa.util import OrderedSet
from jax import make_jaxpr, numpy as jnp
from jax.core import gensym

num_mesh = 2
def fn(*args):
    ret = 0
    for arg in args:
        ret += jnp.mean(jnp.dot(arg, arg))
    return ret
args = [jnp.ones((10, 10)) for _ in range(4)]
jaxpr = make_jaxpr(fn)(*args)

gensym_fn = gensym([jaxpr.jaxpr])
invars = jaxpr.jaxpr.invars
var_mesh = {v: OrderedSet([idx // 2]) for idx, v in enumerate(invars)}
new_jaxpr = ApplyGradRewriter(jaxpr, var_mesh).split_replicated_eqns(gensym_fn, num_mesh)
print(jaxpr)
print(new_jaxpr)

num_stage = 2
donation_mapping = {}
grad_mesh = {k: list(v)[0] for k, v in var_mesh.items()}
outvar_mesh = {}
jaxprs, info = slice_apply_gradient(jaxpr, grad_mesh, outvar_mesh, num_mesh, 2, {})
for jaxpr in jaxprs:
    print(jaxpr)
