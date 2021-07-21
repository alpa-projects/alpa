# TODO (zhuohan): Move this file to tests/

import jax
import jax.numpy as jnp
import numpy as np
import os
import ray
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict as FrozenDictFlax
from jax.experimental.maps import FrozenDict as FrozenDictJax
from parax.model.bert_model import BertConfig, FlaxBertAttention, FlaxBertLayerCollection, FlaxBertLayer, FlaxBertOutput, FlaxBertSelfOutput

from parax import mark_pipeline
from parax import parallelize, set_parallelize_options, DeviceCluster

MB = 1024 ** 2
num_gpus = 2
# in order for ray to work we have to set this
# so the driver program and actor program can share GPUs...
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
jax.config.update('jax_platform_name', 'cpu')
# assert len(jax.local_devices()) >= num_gpus
# devices = tuple(jax.local_devices()[:num_gpus])

def is_sequence(x):
    try:
        iter(x)
    except TypeError:
        return False
    else:
        return True

def assert_allclose(x, y):
    if isinstance(x, dict) or isinstance(x, FrozenDictJax) or isinstance(x, FrozenDictFlax):
        assert isinstance(y, dict) or isinstance(y, FrozenDictJax) or isinstance(x, FrozenDictFlax)
        assert set(x.keys()) == set(y.keys())
        for k in x.keys():
            assert_allclose(x[k], y[k])
    elif is_sequence(x) and not hasattr(x, '__array__'):
        assert is_sequence(y) and not hasattr(y, '__array__')
        assert len(x) == len(y)
        for x_elt, y_elt in zip(x, y):
            assert_allclose(x_elt, y_elt)
    elif hasattr(x, '__array__') or np.isscalar(x):
        assert hasattr(y, '__array__') or np.isscalar(y)
        x = np.asarray(x)
        y = np.asarray(y)
        assert np.allclose(x, y, rtol=1e-3, atol=5e-06), f"{x}, {y}"
    elif x == y:
        return
    else:
        raise TypeError((type(x), type(y)))


class Model(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.layer0 = FlaxBertLayer(config=self.config, dtype=self.dtype)
        self.layer1 = FlaxBertLayer(config=self.config, dtype=self.dtype)

    def __call__(self, x, attention_mask):
        # FIXME (zhuohan): if don't require the gradient of x here, the
        #                  backward pass of the pipeline start will not
        #                  be generated.
        x, = mark_pipeline(x, name='1', mark_type='start')
        layer_outputs = self.layer0(x, attention_mask)
        x = layer_outputs[0]
        x, = mark_pipeline(x, name='1', mark_type='end')
        x, = mark_pipeline(x, name='2', mark_type='start')
        layer_outputs = self.layer1(x, attention_mask)
        x = layer_outputs[0]
        return x

def train_step(optimizer, batch, apply_fn):
    def loss_func(params, x, y, attention_mask):
        out = apply_fn(params, x, attention_mask)
        loss = jnp.mean((out - y) ** 2)
        loss, = mark_pipeline(loss, name='2', mark_type='end')
        return loss

    grad_param, grad_x, grad_mask = jax.grad(loss_func, argnums = (0, 1, 2))(optimizer.target, batch['x'], batch['y'], batch['attention_mask'])
    # FIXME (zhuohan): make the pipeline work with apply_gradient
    # new_optimizer = optimizer.apply_gradient(grad_param)
    return grad_param


ray.init(address="auto", ignore_reinit_error=True)

device_cluster = DeviceCluster()
mesh = device_cluster.get_virtual_mesh()
batch_size = 4
seq_len = 2048
hidden_size = 1024
num_heads = 1024 // 64

x = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)
y = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32) * 23 # * np.arange(hidden_size)[None, None, :]
attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

# Init model and optimizer
model = Model(config=BertConfig(
    hidden_size=hidden_size,
    num_attention_heads=num_heads))
rngkey = jax.random.PRNGKey(0)
params = model.init(rngkey, x, attention_mask)
optimizer = optim.GradientDescent(1e-2).create(params)

gradients = train_step(optimizer, {"x": x, "y": y, "attention_mask": attention_mask}, model.apply)
strategy = "3d_parallel"

set_parallelize_options(devices=mesh, strategy=strategy)
pipelined_train_step = parallelize(train_step)
import time
for i in range(10):
    start = time.time()
    gradients_with_pipeline = pipelined_train_step(optimizer, {"x": x, "y": y, "attention_mask": attention_mask}, model.apply)
    duration = time.time() - start
    print(i, duration)

# print("gradients", gradients)
# print("gradients_with_pipeline", gradients_with_pipeline)
assert_allclose(gradients, gradients_with_pipeline)
