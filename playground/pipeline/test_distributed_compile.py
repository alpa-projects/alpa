from flax import linen as nn, optim
import jax
from jax._src.api import make_jaxpr
from jax.core import gensym
import jax.numpy as jnp
from alpa.mesh_executable import NormalMeshDriverExecutable, ProtoAndSharding
from alpa.pipeline_parallel.apply_grad import compute_grad_to_accumulate_grad
import ray

from alpa import DeviceCluster, manual_layer_slicing, mark_pipeline
from alpa.model.bert_model import BertConfig, FlaxBertLayer
from alpa.pipeline_parallel.stage_profiling import (compile_all,
                                                     generate_stage_info,
                                                     split_global_use_and_donate)
from alpa.pipeline_parallel.three_d_parallel import (
    split_compute_grad_and_apply_grad, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_backward_computation_pipeline_marks)

ray.init(address="auto")
jax.config.update('jax_platform_name', 'cpu')
virtual_mesh = DeviceCluster().get_virtual_physical_mesh()

N = 10


class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layers = [
            FlaxBertLayer(config=self.config, dtype=self.dtype)
            for _ in range(N)
        ]

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
compute_jaxpr, _, _ = split_compute_grad_and_apply_grad(origin_jaxpr)
gensym_fn = gensym([compute_jaxpr.jaxpr])
reduction_vector = [True] * len(compute_jaxpr.jaxpr.outvars)
acc_grad_jaxpr, acc_grad_dict, grad_in_to_out = compute_grad_to_accumulate_grad(
    compute_jaxpr, reduction_vector, gensym_fn)

stages = slice_closed_jaxpr_by_full_pipeline_marks(acc_grad_jaxpr)
stages = mark_missing_vars_in_backward_computation_pipeline_marks(stages,
                                                                  acc_grad_jaxpr.jaxpr.invars,
                                                                  acc_grad_jaxpr.jaxpr.outvars)

donated_global_invars = compute_jaxpr.jaxpr.invars[:-2]
global_invars = acc_grad_jaxpr.jaxpr.invars
global_outvars = acc_grad_jaxpr.jaxpr.outvars
global_donation_mapping = dict()

num_layer_per_stage = 2
stage_infos = []
for start in range(0, N, int(2 * N / num_layer_per_stage)):
    stop = start + num_layer_per_stage
    indices = list(range(start, stop))
    donation_mapping, global_used, new_layers = split_global_use_and_donate(
        stages, indices, global_donation_mapping, global_outvars)
    stage_info = generate_stage_info(stages, indices, donation_mapping,
                                   global_used, str(start))
    stage_infos.append(stage_info)

compiled_outputs = compile_all(stage_infos,
                               virtual_mesh.get_default_logical_mesh(), 16, 4)
physical_mesh = virtual_mesh.get_physical_mesh()
for compiled_output, stage_info in zip(compiled_outputs, stage_infos):
    _, avals, out_avals, tot_donation = stage_info
    proto, config, in_shardings, out_shardings = compiled_output
    compiled = ProtoAndSharding(proto=proto,
                                input_shardings=in_shardings,
                                output_shardings=out_shardings)
    donated_invars = (True,) * len(tot_donation) + (False,) * (
        len(avals) - len(tot_donation))
    executable = NormalMeshDriverExecutable(physical_mesh, compiled, config,
                                            avals, out_avals, donated_invars)
    executable.profile_with_dummy_inputs()
