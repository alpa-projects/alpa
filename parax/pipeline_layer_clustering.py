"""Layer clustering and remat by layer."""
import numba
import numpy as np
import jax
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar, gensym, new_jaxpr_eqn, jaxpr_as_fun
from jax.interpreters import xla
from jax._src.tree_util import tree_flatten
from parax import mark_pipeline_jaxpreqn
from parax.pipeline_stage import JaxPipelineStage

from functools import wraps
from typing import List, Sequence, Callable

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

def clusters_edges_cost(starts : List[List['JaxprEqn']], end : List['JaxprEqn']): # from a list of culsters, to a cluster.
    out_tensors = set()
    for start in starts:
      for eqn in start:
        out_tensors = out_tensors.union(set(eqn.outvars))
    in_tensors = set()
    for eqn in end:
      for invar in eqn.invars:
        if isinstance(invar, Var) and invar in out_tensors:
          in_tensors.add(invar)
    acc = 0
    for in_tensor in in_tensors:
      acc += in_tensor.aval.size
    return acc

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
      acc += in_tensor.aval.size
    return acc

def slice_jaxpr_optimized(jaxpr : Jaxpr, layer_num : int, eps : float):
    length = len(jaxpr.eqns)
    weights = np.array([eqn_flops(eqn) for eqn in jaxpr.eqns])
    layer_weight_upper_bound = np.sum(weights) / layer_num * (1 + eps)
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

    @numba.jit(nopython=True)
    def DP(C):
      A = np.full((length + 1, layer_num + 1), np.inf, dtype=np.float32)
      A_argmin = np.full((length + 1, layer_num + 1), -1, dtype=np.int32)
      B = np.full((length + 1, length +1), np.inf, dtype=np.float32)
      A[0, 0] = 0
      for l in range(1, length + 1):
        tot = 0
        for r in range(l, length + 1):
          tot += weights[r - 1]
          if tot <= layer_weight_upper_bound:
            B[l, r] = 0

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
    assert r == 0
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

def forward(fn : Callable, layer_num : int, eps : float = 0, use_remat : bool = False, remat_ratio : float = 1.0):
    def slice_by_solution(closed_jaxpr : ClosedJaxpr, sliced_eqns) -> Sequence[JaxPipelineStage]:
        n_layers = len(sliced_eqns)
        global_invars = set(closed_jaxpr.jaxpr.invars)
        global_consts_dir = dict(zip(closed_jaxpr.jaxpr.constvars, closed_jaxpr.consts))
        global_outvars = set(var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
        var2stage = {}
        result_stages = []
        current_stage = None
        current_stage_intermediate_vars = set()

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

        for i, eqns in enumerate(sliced_eqns):
          # pipeline start
          current_stage = JaxPipelineStage(name=str(i))
          current_stage_intermediate_vars = set()

          for var in layer_pipeline_invars[i]:
            if not isinstance(var, Literal):
                current_stage.pipeline_invars.add(var)
            if isinstance(var, Literal) or (var in current_stage.pipeline_invars) or (
              var in current_stage_intermediate_vars):
                continue
            if var in global_consts_dir:
              if var not in current_stage.consts_dir:
                current_stage.consts_dir[var] = global_consts_dir[var]
            elif var in global_invars:
              if var not in current_stage.global_invars:
                current_stage.global_invars.add(var)
            else:
              if var not in var2stage:
                raise ValueError("Unknown variable {}".format(var))
              original_stage = var2stage[var]
              if original_stage.name == current_stage.name:
                if var not in original_stage.local_outvars:
                  original_stage.local_outvars.add(var)
                if var not in current_stage.local_invars:
                  current_stage.local_invars.add(var)
              else:
                raise ValueError("Variable {} should be indicated as a pipeline stage input.".format(var))
          # all other eqns
          for eqn in eqns:
            for var in eqn.invars:
              if isinstance(var, Literal) or (var in current_stage.pipeline_invars) or (
                var in current_stage_intermediate_vars):
                continue
              if var in global_consts_dir:
                if var not in current_stage.consts_dir:
                  current_stage.consts_dir[var] = global_consts_dir[var]
              elif var in global_invars:
                if var not in current_stage.global_invars:
                  current_stage.global_invars.add(var)
              else:
                if var not in var2stage:
                  raise ValueError("Unknown variable {}".format(var))
                original_stage = var2stage[var]
                if original_stage.name == current_stage.name:
                  if var not in original_stage.local_outvars:
                    original_stage.local_outvars.add(var)
                  if var not in current_stage.local_invars:
                    current_stage.local_invars.add(var)
                else:
                  raise ValueError("Variable {} should be indicated as a pipeline stage input.".format(var))

            for var in eqn.outvars:
              if not isinstance(var, DropVar):
                current_stage_intermediate_vars.add(var)
                var2stage[var] = current_stage
                if var in global_outvars:
                  current_stage.global_outvars.add(var)

            current_stage.eqns.append(eqn)
          # pipeline end
          current_stage.pipeline_outvars = set(var for var in layer_pipeline_outvars[i] if not isinstance(var, DropVar))
          result_stages.append(current_stage)
          current_stage = None

        for stage in result_stages:
            stage.invars = list(stage.pipeline_invars | stage.global_invars | stage.local_invars)
            stage.outvars = list(stage.pipeline_outvars | stage.global_outvars | stage.local_outvars)

        return result_stages
    ''''''
    if use_remat: 
      @wraps(fn)
      def wrapped(*args):
        origin_jaxpr = jax.make_jaxpr(fn)(*args)
        solution = slice_jaxpr_optimized(origin_jaxpr, layer_num, eps)
        global_invars = origin_jaxpr.jaxpr.invars

        sliced_layers = slice_by_solution(origin_jaxpr, solution)
        sliced_jaxprs = [layer.closed_jaxpr() for layer in sliced_layers]
        sliced_runnables = [jax.remat(jaxpr_as_fun(layer.closed_jaxpr())) for layer in sliced_layers]

        flatten_inputs, _ = tree_flatten(args)
        glob_vars = dict(zip(global_invars, flatten_inputs))
        for (closed_jaxpr, runnable) in zip(sliced_jaxprs, sliced_runnables):
          args = []
          for invar in closed_jaxpr.jaxpr.invars:
            args.append(glob_vars[invar])
          ans = runnable(*args)
          for i, outvar in enumerate(closed_jaxpr.jaxpr.outvars):
            glob_vars[outvar] = ans[i]
        return ans

      return wrapped
    else:
      return fn
