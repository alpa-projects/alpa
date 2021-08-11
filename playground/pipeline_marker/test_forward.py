from timeit import timeit
from typing import Sequence
from functools import partial

import jax
from jax import jit
from jax.lib import xla_client
import jax.numpy as jnp
from parax.pipeline_layer_clustering import forward
from parax.model.bert_model import BertConfig, FlaxBertLayerCollection

def fwd_berts(bert_config : BertConfig, 
                    batch_size : int, seq_len : int):
    hidden_size = bert_config.hidden_size

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    model = FlaxBertLayerCollection(bert_config)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)

    def forward_fn(params, batch):
        rngs = {"dropout": batch["rng"]}
        return model.apply(params,
                          batch["hidden_states"],
                          batch["attention_mask"],
                          rngs=rngs)[0]

    batch = {"hidden_states": hidden_states,
              "attention_mask": attention_mask,
              "label": label,
              "rng": rngkey}
    return forward_fn, (params, batch)

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
origin_fn, inputs = fwd_berts(bert_config, batch_size, seq_len)
wrapped_remat = forward(origin_fn, layer_num=layer_num, use_remat=True)

def wrapped_origin(*args):
  return jit(origin_fn)(*args)

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

remat_out = test(wrapped_remat)
origin_out = test(wrapped_origin)
flatten_remat_out, remat_tree = jax.tree_flatten(remat_out)
flatten_origin_out, origin_tree = jax.tree_flatten(origin_out)
assert remat_tree == origin_tree
for remat_tensor, origin_tensor in zip(flatten_remat_out, flatten_origin_out):
  assert jnp.allclose(remat_tensor, origin_tensor)
''' the below tests performance
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
'''