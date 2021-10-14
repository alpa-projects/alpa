from flax import linen as nn, optim
import jax
from jax._src.api import make_jaxpr
from jax.lib import xla_bridge
import jax.numpy as jnp
import numpy as np
from parax.util import jaxpr_to_hlo_computation
import ray
import multiprocessing

import parax
from parax import (parallelize, global_config, set_parallelize_options,
                   DeviceCluster, manual_layer_slicing)
from parax.model.bert_model import BertConfig, FlaxBertLayer
from parax.pipeline_parallel.primitive_def import mark_pipeline
from parax.pipeline_parallel.three_d_parallel import (
    split_compute_and_apply, slice_closed_jaxpr_by_full_pipeline_marks,
    mark_missing_vars_in_pipeline_marks)
from parax.testing import assert_allclose


class MLP_Model(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        mark_pipeline(name='1', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        mark_pipeline(name='1', mark_type='end')
        mark_pipeline(name='2', mark_type='start')
        x = nn.Dense(features=self.hidden_dim, use_bias=True)(x)
        x = nn.Dense(features=self.output_dim, use_bias=True)(x)
        return x


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

def setUp():
    ray.init(address="auto")
    jax.config.update('jax_platform_name', 'cpu')
    virtual_mesh = DeviceCluster().get_virtual_mesh()
    set_parallelize_options(devices=virtual_mesh, strategy="3d_parallel")

def tearDown():
    ray.shutdown()

CompileWorkerPool = parax.device_mesh.CompileWorkerPool
setUp()

batch_size = 256
hidden_dim = 16
input_dim = output_dim = hidden_dim
model = MLP_Model(hidden_dim=hidden_dim, output_dim=output_dim)
x = jnp.array(np.random.rand(batch_size, input_dim))
y = jnp.array(np.random.rand(batch_size, output_dim))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x)
optimizer = optim.GradientDescent(1e-2).create(params)
batch = {'x': x, 'y': y}

@manual_layer_slicing
def loss_func(params, x, y):
    out = model.apply(params, x)
    loss = jnp.mean((out - y)**2)
    mark_pipeline(name='2', mark_type='end')
    return loss

def train_step(optimizer, batch):
    param_grad = parax.grad(loss_func)(optimizer.target, batch['x'],
                                        batch['y'])
    return param_grad

global_config.num_micro_batches = 1

# copy to prevent from donation
corr = train_step(optimizer, batch)

origin_jaxpr = make_jaxpr(train_step, static_argnums=(2,))(optimizer, batch)

compute_jaxpr, _, _ = split_compute_and_apply(origin_jaxpr)
stages = slice_closed_jaxpr_by_full_pipeline_marks(compute_jaxpr)
stages = mark_missing_vars_in_pipeline_marks(stages, compute_jaxpr.jaxpr.invars,
                                             compute_jaxpr.jaxpr.outvars)
fake_donations = [(False,) * len(stage.invars) for stage in stages]

backend = xla_bridge.get_backend('gpu')
num_cpus = int(DeviceCluster().num_cpus * 0.5)
pool = CompileWorkerPool(num_cpus, sum(DeviceCluster().num_devices))

backup_config = global_config.backup()

virtual_mesh = DeviceCluster().get_virtual_mesh()
global_config.num_micro_batches = None
global_config.devices = virtual_mesh
global_config.strategy = "shard_parallel"
global_config.use_dummy_value_for_benchmarking = True
cur_config = global_config.backup()
w = pool.pop_idle()

name = 'profile_0_shard_parallel'
stage = stages[0]
donate_invars = fake_donations[0]
logical_mesh = virtual_mesh.get_default_logical_mesh()
built = jaxpr_to_hlo_computation(name, stage.closed_jaxpr(), donate_invars, backend)
avals = [var.aval for var in stage.invars]
out_avals = [var.aval for var in stage.outvars]
proto = built.as_serialized_hlo_module_proto()
result = w.compile_single_layer_with_search.remote(cur_config, logical_mesh, proto, avals, out_avals, fake_donations[0])
proto, strategy = ray.get(result)
pool.push(w)
global_config.restore(backup_config)
tearDown()