import numpy as np
from time import time
from flax import linen as nn, optim
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import ray

from alpa import DeviceCluster, manual_layer_slicing, mark_pipeline
from alpa.device_mesh import VirtualPhysicalMesh
from alpa.model.bert_model import BertConfig, FlaxBertLayer
from alpa.pipeline_parallel.three_d_parallel import (
    split_compute_grad_and_apply_grad, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_backward_computation_pipeline_marks)
from alpa.pipeline_parallel.stage_construction import get_submesh_choices, dp, get_sliced_virtual_submeshes, get_compute_cost, get_stage_and_mesh_assignments

ray.init(address="auto")
jax.config.update('jax_platform_name', 'cpu')
virtual_mesh = DeviceCluster().get_virtual_physical_mesh()


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

    @manual_layer_slicing
    def loss_func(params, x, y, attention_mask):
        out = apply_fn(params, x, attention_mask)
        loss = jnp.mean((out - y)**2)
        mark_pipeline(name=str(N - 1), mark_type='end')
        return loss

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
stages = slice_closed_jaxpr_by_full_pipeline_marks(compute_jaxpr)
stages = mark_missing_vars_in_backward_computation_pipeline_marks(stages, compute_jaxpr.jaxpr.invars,
                                                                  compute_jaxpr.jaxpr.outvars)


donation_mapping = {}
global_invars = compute_jaxpr.jaxpr.invars
global_outvars = compute_jaxpr.jaxpr.outvars
all_invars = [set(stage.invars) for stage in stages]
print(compute_jaxpr)
print(all_invars)

virtual_mesh = DeviceCluster().get_virtual_physical_mesh()

submesh_choices = get_submesh_choices(virtual_mesh)

M = len(submesh_choices)
compute_cost = np.full((N, N, M), np.inf)

compute_cost = get_compute_cost(virtual_mesh, submesh_choices, stages, donation_mapping, global_outvars)

print("profiled compute cost", compute_cost)

compute_cost = np.array(
[[[0.00112862, 0.00207896, 0.00304582, 0.00409389, 0.00481757, 0.0058842 , 0.00729934, 0.00901646, 0.01083485, 0.01064126],
 [    np.inf, 0.00105063, 0.00192263, 0.00338936, 0.00393539, 0.00490199, 0.00584266, 0.0072612 , 0.00946384, 0.01016763],
 [    np.inf,     np.inf, 0.00129975, 0.00242482, 0.00291726, 0.00394379, 0.00500327, 0.00620286, 0.0075642 , 0.00776463],
 [    np.inf,     np.inf,     np.inf, 0.00107974, 0.00194375, 0.00296365, 0.00394927, 0.00489317, 0.0060268 , 0.00686378],
 [    np.inf,     np.inf,     np.inf,     np.inf, 0.00113273, 0.00208476, 0.00312124, 0.00414051, 0.00488673, 0.00603056],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00115853, 0.00214725, 0.00309205, 0.00406925, 0.00486824],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.0011634 , 0.00212847, 0.00300874, 0.00403778],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00113964, 0.00209594, 0.00295475],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00112536, 0.00208275],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00113214],],
[[0.0030249 , 0.00583315, 0.00871592, 0.01152415, 0.01424082, 0.01615058, 0.01970495, 0.02182685, 0.02624578, 0.02759846],
 [    np.inf, 0.00283125, 0.00541072, 0.00810671, 0.0113883 , 0.0142146 , 0.01630463, 0.01949045, 0.02265135, 0.02431562],
 [    np.inf,     np.inf, 0.00275834, 0.00543684, 0.00856792, 0.01125206, 0.01419446, 0.01846258, 0.01882169, 0.02256897],
 [    np.inf,     np.inf,     np.inf, 0.00282031, 0.00544018, 0.00806549, 0.01151021, 0.01445823, 0.01596944, 0.01954889],
 [    np.inf,     np.inf,     np.inf,     np.inf, 0.00288251, 0.00546715, 0.00849128, 0.01137638, 0.01331025, 0.01597357],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00281795, 0.00563383, 0.00851236, 0.01133339, 0.01377805],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.0027566 , 0.00544667, 0.00806091, 0.01041269],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00283482, 0.00553597, 0.00840436],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00294116, 0.00520253],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00248777],],
[[0.00318106, 0.00561643, 0.00816067, 0.01074386, 0.01330863, 0.01584069, 0.01861776, 0.02112714, 0.02398107, 0.02674866],
 [    np.inf, 0.00313836, 0.00568464, 0.00836942, 0.01092143, 0.01332755, 0.015868  , 0.01875334, 0.0215208 , 0.02460371],
 [    np.inf,     np.inf, 0.00307181, 0.00560925, 0.00822319, 0.01079559, 0.01324073, 0.0162802 , 0.01885197, 0.02085225],
 [    np.inf,     np.inf,     np.inf, 0.00309396, 0.00569873, 0.00842341, 0.01113261, 0.01343475, 0.01580254, 0.01800921],
 [    np.inf,     np.inf,     np.inf,     np.inf, 0.00313062, 0.00563579, 0.00816891, 0.01091221, 0.01354008, 0.01555475],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00304008, 0.00569354, 0.00829389, 0.01103203, 0.01338752],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00318387, 0.00579458, 0.00826253, 0.01069681],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00314818, 0.00580152, 0.00824009],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00310455, 0.005536  ],
 [    np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf,     np.inf, 0.00285437],]]
).transpose((1, 2, 0))

print(compute_cost.shape, (N, N, M))
print("previously tested compute cost", compute_cost)

cost, solution = dp(N, virtual_mesh.num_devices, batch_size, submesh_choices, compute_cost)
print("-" * 30, "Solution", "-" * 30)
print("Cost:", cost)
print(solution)

sliced_meshes = get_sliced_virtual_submeshes(virtual_mesh, submesh_choices, solution)
print("sliced_meshes", sliced_meshes)

solution, sliced_meshes = get_stage_and_mesh_assignments(virtual_mesh, stages, donation_mapping, global_outvars, batch_size)
print("solution, sliced_meshes", solution, sliced_meshes)

ray.shutdown()
