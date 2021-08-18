"""Layer clustering and remat by layer."""
from jax._src.api import make_jaxpr
import numba
import numpy as np
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar, gensym, new_jaxpr_eqn, jaxpr_as_fun
from jax.interpreters import xla
from parax import mark_pipeline_jaxpreqn

from functools import wraps
from typing import List, Callable

# TODO: different operations takes different time
# e.g. add v.s. pow

gpu_backend = xc.get_local_backend("gpu")

def call_to_xla_computation(eqn : JaxprEqn):
  xe = xc._xla
  prim = eqn.primitive
  backend = gpu_backend

  c = xb.make_computation_builder(f"primitive_computation_{prim.name}")

  name = xla.extend_name_stack(prim.name)

  op_metadata = xla.make_op_metadata(prim, eqn.params)
  c.set_op_metadata(op_metadata)
  xla_args, _ = xla._xla_callable_args(
      c, list(map(lambda x : x.aval, eqn.invars)),
      len(eqn.invars) > 100)
  axis_env = xla.AxisEnv(1, (), ())

  new_params = xla.check_backend_params(eqn.params, backend)
  rule = xla.call_translations[eqn.primitive]
  ans = rule(c, axis_env, xla_args,
              name, backend=backend, **new_params)

  assert isinstance(ans, xe.XlaOp)
  c.clear_op_metadata()
  try:
    return c.build(ans)
  except RuntimeError as e:
    msg = (" ".join(map(str, e.args)) + "\n"
           "This is a bug in JAX's shape-checking rules; please report it!\n"
           "https://github.com/google/jax/issues\n")
    raise RuntimeError(msg) from e

def eqn_flops(eqn : JaxprEqn) -> float:
    if eqn.primitive in xla.call_translations:
      xla_computation = call_to_xla_computation(eqn)
    else:
      xla_computation = xla.primitive_subcomputation(eqn.primitive, *map(lambda x : x.aval, eqn.invars), **eqn.params)
    hlo_module = xla_computation.as_hlo_module()
    properties = xc._xla.hlo_module_cost_analysis(gpu_backend, hlo_module)
    return properties["flops"] if "flops" in properties else 0.0

def cluster_edges_cost(start : List['JaxprEqn'], end : List['JaxprEqn']):
    out_tensors = set()
    for eqn in start:
      out_tensors = out_tensors.union(set(eqn.outvars))
    in_tensors = set()
    for eqn in end:
      for invar in eqn.invars:
        if isinstance(invar, Var) and invar in out_tensors:
          in_tensors.add(invar)
    acc = 0
    for in_tensor in in_tensors:
      acc += in_tensor.aval.size * in_tensor.aval.dtype.itemsize
    return acc

def slice_jaxpr(jaxpr : Jaxpr, layer_num : int, eps : float):
    length = len(jaxpr.eqns)
    weights = np.array([eqn_flops(eqn) for eqn in jaxpr.eqns])
    C = np.full((length + 1, length +1), 0, dtype=np.float32)
    # init

    outvars = set()
    for k in range(0, length + 1):
      if k > 0: outvars = outvars.union(jaxpr.eqns[k - 1].outvars)
      invars = set()
      tot = 0
      for r in range(k + 1, length + 1):
          for invar in jaxpr.eqns[r - 1].invars:
            if isinstance(invar, Var) and invar in outvars\
              and invar not in invars:
              invars.add(invar)
              tot += invar.aval.size
          C[k, r] = tot

    heavy_op_bound = np.sum(weights) * 0.01
    LAYER_HEAVY_OP_BOUND = np.count_nonzero(weights >= heavy_op_bound) / layer_num
    # TODO(yonghao): if LAYER_HEAVY_OP_BOUND <= 2, layer num too large
    LAYER_HEAVY_OP_BOUND = int(max(LAYER_HEAVY_OP_BOUND + 2,
                                   LAYER_HEAVY_OP_BOUND * (1 + eps)))

    @numba.jit(nopython=True)
    def DP(C):
      A = np.full((length + 1, layer_num + 1), np.inf, dtype=np.float32)
      A_argmin = np.full((length + 1, layer_num + 1), -1, dtype=np.int32)
      B = np.full((length + 1, length + 1), np.inf, dtype=np.float32)
      A[0, 0] = 0
      for l in range(1, length + 1):
        cnt = 0
        for r in range(l, length + 1):
          if weights[r - 1] >= heavy_op_bound:
            cnt += 1
          if cnt < 1: continue
          elif cnt <= LAYER_HEAVY_OP_BOUND:
            B[l, r] = 0
          else: break
      for q in range(1, layer_num + 1):
        for r in range(1, length + 1):
          for k in range(0, r):
            new_value = A[k, q - 1] + B[k + 1, r] + C[k, r]
            if new_value < A[r, q]:
              A[r, q] = new_value
              A_argmin[r, q] = k
      return A_argmin

    A_argmin = DP(C)

    reversed_sliced_eqns = []

    r = length
    for q in range(layer_num, 0, -1):
      k = A_argmin[r, q]
      reversed_sliced_eqns.append(jaxpr.eqns[k: r])
      r = k
    assert r == 0, 'no solution for layer clustering' if r == -1 else 'unknown error'
    return list(reversed(reversed_sliced_eqns))

def add_pipeline_markers(closed_jaxpr : ClosedJaxpr, sliced_eqns):
  n_layers = len(sliced_eqns)
  global_invars = set(closed_jaxpr.jaxpr.invars)
  global_consts_dir = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
  layer_pipeline_invars = [set() for _ in range(n_layers)]
  layer_pipeline_outvars = [set() for _ in range(n_layers)]
  var_layer_dict = {}
  for i, eqns in enumerate(sliced_eqns):
    for eqn in eqns:
      for var in eqn.invars:
        if (not isinstance(var, Literal) and var not in global_consts_dir
            and var not in global_invars and var_layer_dict[var] != i):
          layer_pipeline_invars[i].add(var)
          layer_pipeline_outvars[var_layer_dict[var]].add(var)
      for var in eqn.outvars:
        if not isinstance(var, DropVar):
          var_layer_dict[var] = i
  gensym_func = gensym([closed_jaxpr.jaxpr])
  var_mapping = {}
  def get_mapping(var):
    if isinstance(var, Var):
      return var_mapping.get(var, var)
    else:
      return var

  new_eqns = []
  for i, eqns in enumerate(sliced_eqns):
    # pipeline start eqn
    pipeline_start_invars = []
    pipeline_start_outvars = []
    for var in layer_pipeline_invars[i]:
      new_var = gensym_func(var.aval)
      pipeline_start_invars.append(get_mapping(var))
      pipeline_start_outvars.append(new_var)
      var_mapping[var] = new_var
    new_eqns.append(mark_pipeline_jaxpreqn(pipeline_start_invars, pipeline_start_outvars, str(i), 'start'))
    # all other eqns
    for eqn in eqns:
      new_invars = [get_mapping(var) for var in eqn.invars]
      new_eqns.append(new_jaxpr_eqn(new_invars, eqn.outvars, eqn.primitive, eqn.params, eqn.source_info))
    # pipeline end eqn
    pipeline_end_invars = []
    pipeline_end_outvars = []
    for var in layer_pipeline_outvars[i]:
      new_var = gensym_func(var.aval)
      pipeline_end_invars.append(get_mapping(var))
      pipeline_end_outvars.append(new_var)
      var_mapping[var] = new_var
    new_eqns.append(mark_pipeline_jaxpreqn(pipeline_end_invars, pipeline_end_outvars, str(i), 'end'))
  new_jaxpr = Jaxpr(
      closed_jaxpr.jaxpr.constvars,
      closed_jaxpr.jaxpr.invars,
      [get_mapping(var) for var in closed_jaxpr.jaxpr.outvars],
      new_eqns,
      )
  new_closed_jaxpr = ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)
  return new_closed_jaxpr

def forward(fn : Callable, layer_num : int, eps : float = 0.1, use_pipeline = False):
    @wraps(fn)
    def wrapped(*args):
      if use_pipeline:
        origin_jaxpr = make_jaxpr(fn)(*args)
        solution = slice_jaxpr(origin_jaxpr, layer_num, eps)
        new_jaxpr = add_pipeline_markers(origin_jaxpr, solution)
        ans = jaxpr_as_fun(new_jaxpr)(*args)
        return ans if len(ans) != 1 else ans[0]
      else:
        return fn(*args)

    return wrapped
