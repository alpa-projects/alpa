from time import time
from flax import linen as nn, optim
import jax
from jax import tree_flatten
from jax._src.api import make_jaxpr
import jax.numpy as jnp
from parax.util import benchmark_func
import ray

from parax import DeviceCluster, manual_layer_slicing, mark_pipeline
from parax.model.bert_model import BertConfig, FlaxBertLayer
from parax.pipeline_parallel.mesh_slicing import compile_and_profile_layer_cost_c
from parax.pipeline_parallel.three_d_parallel import (
    split_compute_and_apply, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_pipeline_marks)
from parax.pipeline_parallel.mesh_slicing import (
    compile_and_profile_layer_cost_c, split_global_use_and_donate)


ray.init(address="auto")
jax.config.update('jax_platform_name', 'cpu')
virtual_mesh = DeviceCluster().get_virtual_mesh()


N = 10
class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [FlaxBertLayer(config=self.config, dtype=self.dtype) for _ in range(N)]

    def __call__(self, x, attention_mask):
        for i in range(N):
            mark_pipeline(name=str(i), mark_type='start')
            layer_outputs = self.layers[i](x, attention_mask)
            x = layer_outputs[0]
            if i != N - 1:
                mark_pipeline(name=str(i), mark_type='end')
        return x


def train_step(optimizer, batch, apply_fn):

    def loss_func(params, x, y, attention_mask):
        out = apply_fn(params, x, attention_mask)
        loss = jnp.mean((out - y)**2)
        mark_pipeline(name=str(N - 1), mark_type='end')
        return loss

    loss_func = manual_layer_slicing(loss_func)
    grad_param = jax.grad(loss_func)(optimizer.target, batch['x'], batch['y'],
                                     batch['attention_mask'])

    # new_optimizer = optimizer.apply_gradient(grad_param)
    return grad_param


batch_size = 4
seq_len = 64
hidden_size = 256
num_heads = 1
x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
y = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32) * 23
attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

model = BertLayer_Model(config=BertConfig(hidden_size=hidden_size,
                                          intermediate_size=hidden_size * 4,
                                          num_attention_heads=num_heads))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x, attention_mask)
optimizer = optim.GradientDescent(1e-2).create(params)
batch = {"x": x, "y": y, "attention_mask": attention_mask}


origin_jaxpr = make_jaxpr(train_step, static_argnums=(2,))(optimizer, batch,
                                                           model.apply)
compute_jaxpr, _, _ = split_compute_and_apply(origin_jaxpr)
stages = slice_closed_jaxpr_by_full_pipeline_marks(compute_jaxpr)
stages = mark_missing_vars_in_pipeline_marks(stages, compute_jaxpr.jaxpr.invars,
                                             compute_jaxpr.jaxpr.outvars)


donated_global_invars = compute_jaxpr.jaxpr.invars[:-2]
global_invars = compute_jaxpr.jaxpr.invars
global_outvars = compute_jaxpr.jaxpr.outvars
all_invars = [set(stage.invars) for stage in stages]
print(compute_jaxpr)
print(all_invars)

physical_mesh = DeviceCluster().get_physical_mesh()
sharding_specs = dict()

def profile(L = 2):
# L = 2
    costs = []
    indices = list(range(2 * N))
    for start in range(N):
        end = start + L - 1
        if end >= N: break
        layer_collections = stages[start:end + 1] + stages[2 * N - end - 1:2 * N - start]
        layer_indices = indices[start:end + 1] + indices[2 * N - end - 1:2 * N - start]
        donate_invars_list, global_used_list = split_global_use_and_donate(layer_collections, layer_indices, all_invars, donated_global_invars, global_outvars)
        donate_invars_list = [[False for _ in stage.invars] for stage in layer_collections]
        # global_used_list = [[True for _ in stage.outvars] for stage in layer_collections]
        cost, in_specs, out_specs = compile_and_profile_layer_cost_c(layer_collections, physical_mesh, donate_invars_list, global_used_list)
        for layer, index, in_spec, out_spec in zip(layer_collections, layer_indices, in_specs, out_specs):
            for v, s in zip(layer.invars, in_spec):
                if v not in global_invars and s is not None:
                    if v in sharding_specs:
                        found = False
                        for choice in sharding_specs[v]:
                            if s == choice[0]:
                                choice[1].add((L, index))
                                found = True
                                break
                        if not found:
                            sharding_specs[v].append((s, set([(L, index)])))
                    else:
                        sharding_specs[v] = [(s, set([(L, index)]))]
        costs.append(cost)
    return costs

for i in range(1, 9):
    tic = time()
    costs = profile(i)
    toc = time()
    print('=' * 10, i, '=' * 10)
    print(f'profiling all combination of {i} layers takes {toc - tic} seconds')
    print(f'profiled costs are: {costs}')
print('=' * 30)
for v, choices in sharding_specs.items():
    if len(choices) > 1:
        print('-' * 20)
        print(f"{v} has {len(choices)} choices, they are: ")
        for choice in choices:
            print('\t sharding spec:', choice[0])
            print('\t layers:', choice[1])
        print('-' * 20)
    else:
        print(f"{v} has only one choice for: {choices[0]}")

physical_mesh.shutdown()


ray.shutdown()
