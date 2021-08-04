from timeit import timeit
from typing import Sequence
from functools import partial

import jax
from jax import jit, remat
from jax.lib import xla_client
import jax.numpy as jnp
from jax._src.tree_util import tree_flatten
from jax.core import jaxpr_as_fun, ClosedJaxpr, Var, Literal, DropVar
from parax.pipeline_layer_clustering import fwd_berts_jaxpr, slice_jaxpr_optimized
from parax.pipeline_stage import JaxPipelineStage
from parax.model.bert_model import BertConfig

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
    # pipeline start eqn
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


layer_num = 5
batch_size = 64
seq_len = 64
hidden_size = 768
num_heads = 768 // 96
bert_config = BertConfig(
    num_hidden_layers=layer_num,
    hidden_size=hidden_size,
    intermediate_size=hidden_size * 4,
    num_attention_heads=num_heads)
origin_jaxpr, inputs = fwd_berts_jaxpr(bert_config, batch_size, seq_len)

solution = slice_jaxpr_optimized(origin_jaxpr, layer_num, 0)

global_invars = origin_jaxpr.jaxpr.invars
flatten_inputs, tree = tree_flatten(inputs)
var_dict = dict(zip(global_invars, flatten_inputs))

sliced_layers = slice_by_solution(origin_jaxpr, solution)
sliced_jaxprs = [layer.closed_jaxpr() for layer in sliced_layers]
sliced_runnables = [jit(remat(jaxpr_as_fun(layer.closed_jaxpr()))) for layer in sliced_layers]

def fn(glob_vars):
  for (closed_jaxpr, runnable) in zip(sliced_jaxprs, sliced_runnables):
    args = []
    for invar in closed_jaxpr.jaxpr.invars:
      args.append(glob_vars[invar])
    ans = runnable(*args)
    for i, outvar in enumerate(closed_jaxpr.jaxpr.outvars):
      glob_vars[outvar] = ans[i]
  return ans

# without jit and remat, the jaxpr of fn is the same as origin_jaxpr

def wrapped_remat(*args):
  flatten_inputs, _ = tree_flatten(args)
  var_dict = dict(zip(global_invars, flatten_inputs))
  return fn(var_dict)
def wrapped_origin(*args):
  flatten_inputs, _ = tree_flatten(args)
  return jit(jaxpr_as_fun(origin_jaxpr))(*flatten_inputs)

new_ans = fn(var_dict)[0]
loss = jnp.mean(new_ans - inputs[1]["label"])

def bypass(params, batch, fn):
  def loss_fn(params):
    output = fn(params, batch)[0]
    return jnp.mean(output - batch["label"])
  
  gradient = jax.grad(loss_fn)(params)
  return gradient

jit_bypass = jit(bypass, static_argnums=(2,))
params, batch = inputs
partial_fn = partial(jit_bypass, params, batch)

def test(apply_fn):
  output = partial_fn(apply_fn)
  jax.tree_util.tree_leaves(output)[0].block_until_ready()
  return output

def test_fn(fn_name):
  stmt = "test({})".format(fn_name)
  number = 30
  timeit(stmt, globals={**globals(), **locals()}, number=5)
  return timeit(stmt, globals={**globals(), **locals()},
                    number=number) / number

remat_cost = test_fn("wrapped_remat")
no_remat_cost = test_fn("wrapped_origin")
print("remat takes {}s, no remat takes {}s, speed is {}% of origin ".format(remat_cost, no_remat_cost, no_remat_cost / remat_cost * 100))

def get_mem(fn):
  partial_bypass = partial(bypass, fn=fn)
  c = jax.xla_computation(partial_bypass)(params, batch)
  gpu_backend = xla_client.get_local_backend("gpu")
  compiled_computation = gpu_backend.compile(c)
  return compiled_computation.total_allocation_size()

remat_mem = get_mem(wrapped_remat)
normal_mem = get_mem(wrapped_origin)
MB = 1024 ** 2
print("remat takes {}MB, no remat takes {}MB, memory is {}% of origin".format(remat_mem / MB, normal_mem / MB, remat_mem / normal_mem * 100))