from math import log2
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from jax._src import ad_util
from jax.core import ClosedJaxpr, JaxprEqn, Jaxpr, Var, Literal, DropVar
from jax.interpreters import xla
from flax import optim
from parax.model.bert_model import BertConfig, FlaxBertLayer, FlaxBertLayerCollection
from typing import List, Union

# TODO: different operations takes different time
# e.g. add v.s. pow
class Estimator:
    def __init__(self):
        self.pointwise_prims_ = (
          lax.neg_p, lax.sign_p,
          lax.floor_p, lax.ceil_p, lax.round_p, lax.is_finite_p,
          lax.exp_p, lax.log_p,
          lax.pow_p,
          lax.tanh_p, lax.tan_p, lax.sin_p, lax.cos_p,
          lax.asin_p, lax.acos_p, lax.atan_p, lax.atan2_p,
          lax.sinh_p, lax.cosh_p, lax.asinh_p, lax.acosh_p, lax.tanh_p,
          lax.add_p, lax.sub_p, lax.abs_p, lax.sqrt_p,
          lax.not_p, lax.and_p, lax.or_p, lax.xor_p, lax.mul_p, lax.div_p,
          lax.max_p, lax.min_p,
          lax.shift_left_p,
          lax.shift_right_arithmetic_p, lax.shift_right_logical_p,
          lax.eq_p, lax.ne_p, lax.ge_p, lax.gt_p, lax.le_p, lax.lt_p,
          lax.convert_element_type_p, lax.bitcast_convert_type_p,
          lax.erf_p,
          lax.transpose_p,
          ad_util.add_jaxvals_p)

        self.unhandled_prims_ = (
          lax.nextafter_p,
          lax.expm1_p, lax.log1p_p,
          lax.regularized_incomplete_beta_p,
          lax.lgamma_p, lax.digamma_p, lax.igamma_p,
          lax.igamma_grad_a_p, lax.igammac_p, lax.random_gamma_grad_p,
          lax.bessel_i0e_p, lax.bessel_i1e_p,
          lax.erfc_p, lax.erf_inv_p,
          lax.real_p, lax.imag_p, lax.complex_p,
          lax.conj_p, lax.population_count_p,
          lax.conv_general_dilated_p,
          lax.squeeze_p)# missing: clz_p(?)

        # TODO: consider dtype
        self.elementwise_flops_ = lambda eqn : max(map(lambda var : var.aval.size, eqn.invars))
        self.default_fn_ = lambda _ : 0

        self.prim_flops_ = {
          lax.dot_general_p: self._dot_flops,
          xla.xla_call_p : self._xla_call_flops,
          lax.rsqrt_p: lambda eqn : 2 * self.elementwise_flops_(eqn),
          lax.integer_pow_p:
            lambda eqn : log2(abs(eqn.params['y'])) * self.elementwise_flops_(eqn),
          lax.broadcast_in_dim_p:
            lambda eqn : eqn.outvars[0].aval.size - eqn.invars[0].aval.size,
          lax.broadcast_p:
            lambda eqn : eqn.outvars[0].aval.size - eqn.invars[0].aval.size,
          lax.pad_p: self.elementwise_flops_,
          lax.reduce_sum_p: self.elementwise_flops_,
          lax.reduce_max_p: self.elementwise_flops_,
          lax.reshape_p: self.default_fn_,
          ad_util.stop_gradient_p: self.default_fn_,
          lax.select_p: self.default_fn_,
          lax.slice_p: self.default_fn_
        }

        self.todo_prims_ = (lax.clamp_p, lax.concatenate_p)

    def _dot_flops(self, eqn : JaxprEqn) -> int:
        dims = eqn.params['dimension_numbers']
        contract_dims = dims[0]
        batch_dims = dims[1]
        acc = 1
        for i, var in enumerate(eqn.invars):
          contract_dim = contract_dims[i]
          batch_dim = batch_dims[i]
          shape = var.aval.shape
          for j, size in enumerate(shape):
            if j not in contract_dim and j not in batch_dim: acc *= size
        for var_dim in contract_dims[0]:
          shape = eqn.invars[0].aval.shape
          acc *= shape[var_dim]
        for var_dim in batch_dims[0]:
          shape = eqn.invars[0].aval.shape
          acc *= shape[var_dim]
        return acc

    def _xla_call_flops(self, eqn : JaxprEqn) -> int:
        called_jaxpr = eqn.params['call_jaxpr']
        acc = 0
        for eqn in called_jaxpr.eqns:
          acc += self.eqn_flops(eqn)
        return acc

    def eqn_flops(self, eqn : JaxprEqn) -> int:
        if eqn.primitive in self.pointwise_prims_:
          return self.elementwise_flops_(eqn)
        if eqn.primitive in self.unhandled_prims_:
          return self.default_fn_(eqn)
        if eqn.primitive in self.prim_flops_:
          return self.prim_flops_[eqn.primitive](eqn)
        raise Exception("unimplemented")  # TODO: or use default?

    def analyze_prims(self, jaxpr):
        unhandled = []
        todo = []
        unconsidered = []
        for eqn in jaxpr.eqns:
          if eqn.primitive in self.unhandled_prims_:
            unhandled.append(eqn.primitive)
          elif eqn.primitive in self.todo_prims_:
            todo.append(eqn.primitive)
          elif eqn.primitive in self.prim_flops_:
            self.prim_flops_[eqn.primitive](eqn)
          elif eqn.primitive in self.pointwise_prims_:
            self.elementwise_flops_(eqn)
          else:
            unconsidered.append(eqn.primitive)
        unhandled = set(unhandled)
        todo = set(todo)
        unconsidered = set(unconsidered)
        if unhandled: print(unhandled, "are unhandled")
        if todo: print(todo, "are in todo list")
        if unconsidered: print(unconsidered, "are not considered")

    def clusters_edges_cost(self, starts : List[List['JaxprEqn']], end : List['JaxprEqn']): # from a list of culsters, to a cluster.
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

    def cluster_edges_cost(self, start : List['JaxprEqn'], end : List['JaxprEqn']):
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

estimator = Estimator()

def SliceJaxpr(jaxpr : Jaxpr, layer_num : int, eps : float):
    length = len(jaxpr.eqns)
    weights = [estimator.eqn_flops(eqn) for eqn in jaxpr.eqns]
    layer_weight_upper_bound = float(sum(weights)) / layer_num * (1 + eps)
    solutions = [
      [[(-1,None)] * layer_num for _ in range(length)]
      for _ in range(length)
    ]
    # init
    for i in range(length):
      for j in range(length):
        if (sum(weights[i : j + 1]) <= layer_weight_upper_bound):
            solutions[i][j][0] = (0,None)

    for num in range(1, layer_num):
      for l in range(length - 1):
        for r in range(l + 1, length):
          for k in range(l, r):
            if solutions[l][k][num-1][0] != -1 and\
              solutions[k + 1][r][0][0] != -1:
              prior = solutions[l][r][num][0]
              current = solutions[l][k][num - 1][0] + \
                  solutions[k + 1][r][0][0] + \
                  estimator.cluster_edges_cost(jaxpr.eqns[l : k + 1],
                    jaxpr.eqns[k + 1 : r + 1])
              if prior == -1 or prior > current:
                solutions[l][r][num] = (current, k)
    return solutions

def slice_jaxpr_optimized(jaxpr : Jaxpr, layer_num : int, eps : float):
    length = len(jaxpr.eqns)
    weights = np.array([estimator.eqn_flops(eqn) for eqn in jaxpr.eqns])
    layer_weight_upper_bound = np.sum(weights) / layer_num * (1 + eps)
    A = np.full((length + 1, layer_num + 1), np.inf, dtype=np.float)
    A_argmin = np.full((length + 1, layer_num + 1), -1, dtype=np.int)
    B = np.full((length + 1, length +1), np.inf, dtype=np.float)
    C = np.full((length + 1, length +1), 0, dtype=np.float)
    # init
    # FIXME (zhuohan): The initialization is O(n^3) right now. It should be optimized to O(n^2).
    A[0, 0] = 0

    for l in range(1, length + 1):
      for r in range(l, length + 1):
        if (sum(weights[l - 1 : r]) <= layer_weight_upper_bound):
          B[l, r] = 0

    for k in range(0, length + 1):
      for r in range(k + 1, length + 1):
          C[k, r] = estimator.cluster_edges_cost(jaxpr.eqns[0 : k], jaxpr.eqns[k : r])

    for q in range(1, layer_num + 1):
      for r in range(1, length + 1):
        for k in range(0, r):
          new_value = A[k, q - 1] + B[k + 1, r] + C[k, r]
          if new_value < A[r, q]:
            A[r, q] = new_value
            A_argmin[r, q] = k

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
  global_outvars = set(var for var in closed_jaxpr.jaxpr.outvars if isinstance(var, Var))
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
  print("layer_pipeline_invars", layer_pipeline_invars)
  print("layer_pipeline_outvars", layer_pipeline_outvars)


if __name__ == "__main__":
    estimator.analyze_prims(bert_layer_jaxpr())
    # SliceJaxpr(fwd_berts_jaxpr(2), 2)
    # Depth = 4
    # jaxpr = edge_weight_test_jaxpr(Depth)
    # from_sets = [[jaxpr.eqns[i],] for i in range(2 ** (Depth - 1))]
    # to_set = [jaxpr.eqns[i + 2 ** (Depth - 1)] for i in range(2 ** (Depth - 1))]
    # print(estimator.clusters_edges_cost(from_sets, to_set))
    layer_num = 2
    closed_jaxpr = fwd_berts_jaxpr(layer_num)
    eqns = closed_jaxpr.eqns
    eqn_num = len(eqns)
    edge_cost = estimator.cluster_edges_cost(
                    eqns[0:int(eqn_num / layer_num)],
                    eqns[int(eqn_num / layer_num) : eqn_num])
    solutions = slice_jaxpr_optimized(closed_jaxpr, layer_num, 0)
    print(solutions)
    add_pipeline_markers(closed_jaxpr, solutions)
    # assert solutions[0][eqn_num - 1][layer_num - 1][1] == int(eqn_num / layer_num - 1)
    # assert solutions[0][eqn_num - 1][layer_num - 1][0] == edge_cost
    # print("success!")
