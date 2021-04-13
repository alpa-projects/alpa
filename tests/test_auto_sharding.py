from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict, freeze
from transformers.models.bert.modeling_flax_bert import FlaxBertAttention

from paranum import parallelize, global_config, testing


MB = 1024 ** 2

def test_donate_buffer():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    @parallelize(donate_argnums=(0,),
                 memory_budget_per_device=3 * MB,
                 devices=devices)
    def add_one(x):
        x = x + 1
        return x

    a = jnp.ones((1024, 1024))
    b = add_one(a)

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    # assert a and b are split over the second dimention
    assert "parameter(0), sharding={devices=[1,4]0,1,2,3}" in hlo_ir
    assert "(param: f32[1024,256]) -> (f32[1024,256])" in hlo_ir


def test_2_layer_mlp():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    class Model(nn.Module):
        hidden_dim: int
        output_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
            x = nn.relu(x)
            x = nn.Dense(features=self.output_dim, use_bias=False)(x)
            return x

    @parallelize(memory_budget_per_device=50 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    batch_size = 128
    hidden_dim = 2048
    input_dim = output_dim = hidden_dim

    x = jnp.ones((batch_size, input_dim))
    y = jnp.ones((batch_size, output_dim))

    model = Model(hidden_dim=hidden_dim, output_dim=output_dim)
    params = FrozenDict({
        "params": {
            "Dense_0": {
                "kernel": jnp.ones((input_dim, hidden_dim)),
            },
            "Dense_1": {
                "kernel": jnp.ones((hidden_dim, output_dim)),
            }
        }
    })
    optimizer = optim.GradientDescent(1e-2).create(params)

    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    # The function should contain only one communication primitive,
    # which is an all-reduce
    assert hlo_ir.count("channel_id") == 1
    assert hlo_ir.count("all-reduce(") == 1
    weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
    weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
    assert isinstance(weight0, pxla.ShardedDeviceArray)
    assert isinstance(weight1, pxla.ShardedDeviceArray)
    # column partitioned
    assert weight0.sharding_spec == pxla.ShardingSpec(
        sharding=(pxla.Chunked([1]), pxla.Chunked([4])),
        mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1)),
    )
    # row partitioned
    assert weight1.sharding_spec == pxla.ShardingSpec(
        sharding=(pxla.Chunked([4]), pxla.Chunked([1])),
        mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1)),
    )


def test_n_layer_mlp():
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    class Model(nn.Module):
        hidden_dim: int
        output_dim: int
        num_layers: int

        @nn.compact
        def __call__(self, x):
            for i in range(self.num_layers-1):
                x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
                x = nn.relu(x)
            x = nn.Dense(features=self.output_dim, use_bias=False)(x)
            return x

    @parallelize(memory_budget_per_device=80 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    batch_size = 128
    hidden_dim = 2048
    num_layers = 6
    input_dim = output_dim = hidden_dim

    x = jnp.ones((batch_size, input_dim))
    y = jnp.ones((batch_size, output_dim))

    model = Model(num_layers=num_layers,
                  hidden_dim=hidden_dim, output_dim=output_dim)
    params = ({
        "params": {}
    })
    for i in range(num_layers):
        if i == 0:
            params['params'][f'Dense_{i}'] = {
                "kernel": jnp.ones((input_dim, hidden_dim))
            }
        elif i == num_layers - 1:
            params['params'][f'Dense_{i}'] = {
                "kernel": jnp.ones((hidden_dim, output_dim))
            }
        else:
            params['params'][f'Dense_{i}'] = {
                "kernel": jnp.ones((hidden_dim, hidden_dim))
            }
    params = FrozenDict(params)

    optimizer = optim.GradientDescent(1e-2).create(params)

    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)


    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    # The function should contain 5 all-reduce
    assert hlo_ir.count("channel_id") == 5
    assert hlo_ir.count("all-reduce(") == 5

    for i in range(num_layers):
        weight = optimizer.target["params"][f"Dense_{i}"]["kernel"]
        assert isinstance(weight, pxla.ShardedDeviceArray)
        if i % 2 == 0:
            # column partitioned
            assert weight.sharding_spec == pxla.ShardingSpec(
                sharding=(pxla.Chunked([1]), pxla.Chunked([4])),
                mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1)),
            )
        else:
            # row partitioned
            assert weight.sharding_spec == pxla.ShardingSpec(
                sharding=(pxla.Chunked([4]), pxla.Chunked([1])),
                mesh_mapping=(pxla.ShardedAxis(0), pxla.ShardedAxis(1)),
            )


if __name__ == "__main__":
    global_config.set_shard_parallel_strategy('auto_sharding')

    test_donate_buffer()
    test_2_layer_mlp()
    test_n_layer_mlp()

