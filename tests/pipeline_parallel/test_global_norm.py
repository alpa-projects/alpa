# FIXME(yonghao): make it a real test
from alpa.pipeline_parallel.apply_grad import split_replicated_eqns
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
new_eqns = split_replicated_eqns(jaxpr.eqns, var_mesh, gensym_fn, num_mesh)
print(jaxpr)
for eqn in new_eqns:
    print(eqn)