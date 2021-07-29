from math import log2
import jax
import jax.numpy as jnp
from jax import lax, ad_util
from jax.core import JaxprEqn
from jax.interpreters import xla
from flax import optim
from parax.model.bert_model import BertConfig, FlaxBertLayer

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
        raise Exception("unimplemented")

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
          if eqn.primitive is xla.xla_call_p:
            print(eqn, eqn.invars[0].aval.shape, eqn.invars[1].aval.shape)
            print(self.prim_flops_[eqn.primitive](eqn))
            raise Exception()
        unhandled = set(unhandled)
        todo = set(todo)
        unconsidered = set(unconsidered)
        if unhandled: print(unhandled, "are unhandled")
        if todo: print(todo, "are in todo list")
        if unconsidered: print(unconsidered, "are not considered")

    def edge_cost(self, eqn1 : JaxprEqn, eqn2 : JaxprEqn) -> int:
      pass

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

estimator = Estimator()

if __name__ == "__main__": 
    jaxpr = bert_layer_jaxpr()
    estimator.analyze_prims(jaxpr)

