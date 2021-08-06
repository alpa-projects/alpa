import numba
import numpy as np
import jax
import jax.numpy as jnp
from jax.lib import xla_client as xc, xla_bridge as xb
from jax.core import ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar, gensym, new_jaxpr_eqn
from jax.interpreters import xla
from flax import optim
from parax import mark_pipeline_jaxpreqn
from parax.model.bert_model import BertConfig, FlaxBertLayer, FlaxBertLayerCollection
from typing import List

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

def bert_layer_jaxpr():
    batch_size = 64
    seq_len = 64
    hidden_size = 768

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    # Init model and optimizer
    model = FlaxBertLayer(BertConfig(
        hidden_size=hidden_size
    ))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
    optimizer = optim.GradientDescent(1e-2).create(params)

    def train_step(optimizer, batch):
        def loss_func(params):
            rngs = {"dropout": batch["rng"]}
            out = model.apply(params,
                              batch["hidden_states"],
                              batch["attention_mask"],
                              rngs=rngs)[0]
            return jnp.mean((out - batch["label"]) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    jaxpr = jax.make_jaxpr(train_step)(optimizer,
                            {"hidden_states": hidden_states,
                            "attention_mask": attention_mask,
                            "label": label,
                            "rng": rngkey})
    return jaxpr

def edge_weight_test_jaxpr(Depth : int):
    inputs = []
    N = 1024
    for _ in range(2 ** Depth):
      inputs.append(jnp.ones((N, N)))
    def computation(inputs):
        for d in range(Depth):
          outputs = []
          for i in range(2 ** (Depth - d - 1)):
            outputs.append(inputs[i * 2] + inputs[i * 2 + 1])
          inputs = outputs
        return inputs[0].mean()

    return jax.make_jaxpr(computation)(inputs)

def fwd_berts_jaxpr(num_layers : int = 2):
    batch_size = 64
    seq_len = 64
    hidden_size = 768
    num_heads = 768 // 96

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    model = FlaxBertLayerCollection(BertConfig(
        num_hidden_layers=num_layers,
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=num_heads))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
    optimizer = optim.GradientDescent(1e-2).create(params)

    def forward(batch):
        rngs = {"dropout": batch["rng"]}
        return model.apply(params,
                          batch["hidden_states"],
                          batch["attention_mask"],
                          rngs=rngs)[0]


    jaxpr = jax.make_jaxpr(forward)({"hidden_states": hidden_states,
                            "attention_mask": attention_mask,
                            "label": label,
                            "rng": rngkey})
    return jaxpr

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


if __name__ == "__main__":
    layer_num = 2
    closed_jaxpr = fwd_berts_jaxpr(layer_num)
    eqns = closed_jaxpr.eqns
    eqn_num = len(eqns)
    edge_cost = cluster_edges_cost(
                    eqns[0:int(eqn_num / layer_num)],
                    eqns[int(eqn_num / layer_num) : eqn_num])
    solutions = slice_jaxpr_optimized(closed_jaxpr, layer_num, 0)
    new_closed_jaxpr = add_pipeline_markers(closed_jaxpr, solutions)
    print(new_closed_jaxpr)
    '''---testing compilation time---'''
    # from timeit import timeit
    # def process(closed_jaxpr, layer_num):
    #   solutions = slice_jaxpr_optimized(closed_jaxpr, layer_num, 0)
    #   return add_pipeline_markers(closed_jaxpr, solutions)
    # layer_num = 70
    # print('building jaxpr...')
    # closed_jaxpr = fwd_berts_jaxpr(layer_num)
    # stmt = 'process(closed_jaxpr, layer_num)'
    # print('testing...')
    # number = 1
    # print(timeit(stmt, globals={**globals(), **locals()}, number=number) / number)