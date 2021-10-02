from flax import linen as nn, optim
import jax
from jax._src.api import make_jaxpr
import jax.numpy as jnp
import ray

from parax import DeviceCluster, manual_layer_slicing, mark_pipeline
from parax.model.bert_model import BertConfig, FlaxBertLayer


class BertLayer_Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        mark_pipeline(name='1', mark_type='start')
        layer_outputs = self.layer0(x, attention_mask)
        x = layer_outputs[0]
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        layer_outputs = self.layer1(x, attention_mask)
        x = layer_outputs[0]
        return x


ray.init(address="auto")
jax.config.update('jax_platform_name', 'cpu')
virtual_mesh = DeviceCluster().get_virtual_mesh()


def train_step(optimizer, batch, apply_fn):

    def loss_func(params, x, y, attention_mask):
        out = apply_fn(params, x, attention_mask)
        loss = jnp.mean((out - y)**2)
        mark_pipeline(name='2', mark_type='end')
        return loss

    loss_func = manual_layer_slicing(loss_func)
    grad_param = jax.grad(loss_func)(optimizer.target, batch['x'], batch['y'],
                                     batch['attention_mask'])

    # new_optimizer = optimizer.apply_gradient(grad_param)
    return grad_param


Inc = 1
batch_size = 2 * Inc
seq_len = 64 * Inc
hidden_size = 256 * Inc
num_heads = 1

x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
y = jnp.ones((batch_size, seq_len, hidden_size),
             dtype=jnp.float32) * 23  # * np.arange(hidden_size)[None, None, :]
attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.float32)

# Init model and optimizer
model = BertLayer_Model(config=BertConfig(hidden_size=hidden_size,
                                          intermediate_size=hidden_size * 4,
                                          num_attention_heads=num_heads))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x, attention_mask)
optimizer = optim.GradientDescent(1e-2).create(params)
batch = {"x": x, "y": y, "attention_mask": attention_mask}

# global_config.num_micro_batches = 2

origin_jaxpr = make_jaxpr(train_step, static_argnums=(2,))(optimizer, batch,
                                                           model.apply)

def dummy_large_trans(*args):
    @manual_layer_slicing
    def dummy_fwd(x, y, z, tgt):
        mark_pipeline(name='1', mark_type='start')
        out = x @ y
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        out = out @ z
        loss = jnp.mean((out - tgt) ** 2)
        mark_pipeline(name='2', mark_type='end')
        return loss
    grad = jax.grad(dummy_fwd)(*args)
    return grad
N = 16384
args = [jnp.zeros((N, N)) for _ in range(4)]

origin_jaxpr = make_jaxpr(dummy_large_trans)(*args)

from parax.pipeline_parallel.three_d_parallel import (
    split_compute_and_apply, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_pipeline_marks)
from parax.pipeline_parallel.mesh_slicing import (
    compile_and_profile_layer_cost_c, create_collective_group,
    profile_layer_cost_e)

compute_jaxpr, _, _ = split_compute_and_apply(origin_jaxpr)
stages = slice_closed_jaxpr_by_full_pipeline_marks(compute_jaxpr)
stages = mark_missing_vars_in_pipeline_marks(stages, compute_jaxpr.jaxpr.invars,
                                             compute_jaxpr.jaxpr.outvars)
# for stage in stages:
#     print(stage.closed_jaxpr())
'''----------------profile cost c----------------'''
# round = 1
# physical_mesh = DeviceCluster().get_physical_mesh()
# tn = "compute1"
# timers(tn).start()
# for t in range(round):
#     print(compile_and_profile_layer_cost_c((stages[0], stages[3]), physical_mesh)[0])
# timers(tn).stop()
# print(timers(tn).elapsed())
# tn = "compute2"
# timers(tn).start()
# for t in range(round):
#     print(compile_and_profile_layer_cost_c((stages[1], stages[2]), physical_mesh)[0])
# timers(tn).stop()
# print(timers(tn).elapsed())
'''----------------profile cost e----------------'''
src = stages[0]
dst = stages[1]
src_mesh = virtual_mesh.slice(1, [[0, 1]])
src_phy_mesh = src_mesh.get_physical_mesh()
dst_mesh = virtual_mesh.slice(1, [[2, 3]])
dst_phy_mesh = dst_mesh.get_physical_mesh()


def all_false_donation(stages):
    return [[False for _ in stage.invars] for stage in stages]


def all_true_outvar(stages):
    return [[True for _ in stage.outvars] for stage in stages]


test_stages = (stages[0], stages[3])
cost_c1, _, out_spec = compile_and_profile_layer_cost_c(
    test_stages, src_phy_mesh, all_false_donation(test_stages),
    all_true_outvar(test_stages))
test_stages = (stages[1], stages[2])
cost_c2, in_spec, _ = compile_and_profile_layer_cost_c(
    test_stages, dst_phy_mesh, all_false_donation(test_stages),
    all_true_outvar(test_stages))

# print(cost_c1, cost_c2)
src_phy_mesh.sync_workers()
dst_phy_mesh.sync_workers()
collective_group = create_collective_group(src_phy_mesh, dst_phy_mesh)

cost_e = profile_layer_cost_e(stages[0], stages[1], out_spec[0], in_spec[0],
                              src_mesh, dst_mesh, collective_group)

print(cost_e)
# collective_group.destroy()
src_phy_mesh.shutdown()
dst_phy_mesh.shutdown()
ray.shutdown()

# LnkCap: Port #2, Speed 8GT/s, Width x16, ASPM not supported, Exit Latency L0s <512ns, L1 <4us
# LnkSta: Speed 2.5GT/s, Width x8, TrErr- Train- SlotClk+ DLActive- BWMgmt- ABWMgmt-