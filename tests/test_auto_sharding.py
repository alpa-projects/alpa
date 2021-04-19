from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.interpreters.pxla import Chunked, ShardedAxis
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict, freeze
from transformers.models.bert.modeling_flax_bert import FlaxBertAttention, FlaxBertLayer

from parax import parallelize, global_config, testing


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

    # assert a and b are split over the second dimension
    assert b.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([1]), Chunked([4])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1))) or\
           b.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([4]), Chunked([1])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)))


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
    assert hlo_ir.count("channel_id") == 1, hlo_ir.count("channel_id")
    assert hlo_ir.count("all-reduce(") == 1
    weight0 = optimizer.target["params"]["Dense_0"]["kernel"]
    weight1 = optimizer.target["params"]["Dense_1"]["kernel"]
    assert isinstance(weight0, pxla.ShardedDeviceArray)
    assert isinstance(weight1, pxla.ShardedDeviceArray)
    # column partitioned
    assert weight0.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([1]), Chunked([4])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
    )
    # row partitioned
    assert weight1.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([4]), Chunked([1])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
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
                sharding=(Chunked([1]), Chunked([4])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )
        else:
            # row partitioned
            assert weight.sharding_spec == pxla.ShardingSpec(
                sharding=(Chunked([4]), Chunked([1])),
                mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
            )


def test_attention(mode='forward'):
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    class Model(nn.Module):
        num_heads: int
        head_size: int
        kernel_init_scale: float = 0.2
        dropout_rate: float = 0.0
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, hidden_states, attention_mask, deterministic: bool=True):
            attention = FlaxBertAttention(
                self.num_heads,
                self.head_size,
                kernel_init_scale=self.kernel_init_scale,
                dropout_rate=self.dropout_rate,
                name="attention",
                dtype=self.dtype,
            )(hidden_states, attention_mask, deterministic=deterministic)
            return attention

    @parallelize(memory_budget_per_device=100 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            rngs = {"dropout": batch['rng']}
            out = apply_fn(params, batch['hidden_states'],
                           batch['attention_mask'], deterministic,
                           rngs=rngs)
            return jnp.mean((out - batch['label']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    @parallelize(memory_budget_per_device=80 * (1 << 20),
                 devices=devices)
    def forward_step(optimizer, batch, apply_fn):
        rngs = {"dropout": batch['rng']}
        out = apply_fn(optimizer.target, batch['hidden_states'],
                       batch['attention_mask'], deterministic,
                       rngs=rngs)
        return out

    batch_size = 4
    seq_len = 128
    hidden_dim = 2048
    num_heads = 16
    per_head = hidden_dim // num_heads
    dropout_rate = 0.0
    deterministic = False

    hidden_states = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    model = Model(num_heads=num_heads, head_size=hidden_dim, dropout_rate=dropout_rate)
    rngkey = jax.random.PRNGKey(0)
    params = FrozenDict({
        "params": {
            "attention": {
                "self": {
                    "query": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "key": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "value": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "out": {
                        "kernel": jnp.ones((num_heads, per_head, hidden_dim)),
                        "bias": jnp.ones((hidden_dim,)),
                    },
                },
                "layer_norm": {
                    "beta": jnp.ones((hidden_dim,)),
                    "gamma": jnp.ones((hidden_dim,)),
                },
            },
        },
    })

    optimizer = optim.GradientDescent(1e-2).create(params)

    if mode == 'forward':
        optimizer = forward_step(optimizer,
                                 {"hidden_states": hidden_states,
                                  "attention_mask": attention_mask,
                                  "label": label,
                                  "rng": rngkey},
                                 model.apply)
    else:
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               model.apply)

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    # The function should contain only one communication primitive,
    # which is an all-reduce
    assert hlo_ir.count("channel_id") == 1
    assert hlo_ir.count("all-reduce(") == 1
    # all weight tensors should be split over the head dimension
    for name in ["query", "key", "value"]:
        weight_q = optimizer.target["params"]["attention"]["self"][name]["kernel"]
        assert weight_q.sharding_spec == pxla.ShardingSpec(
            sharding=(Chunked([1]), Chunked([4]), Chunked([1])),
            mesh_mapping=(ShardedAxis(0), ShardedAxis(1), ShardedAxis(2)),
        )


def test_bert_layer(mode='forward'):
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    class Model(nn.Module):
        num_heads: int
        head_size: int
        intermediate_size: int
        dropout_rate: float = 0.0
        kernel_init_scale: float = 0.2
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, hidden_states, attention_mask, deterministic: bool=True):
            attention = FlaxBertLayer(
                self.num_heads,
                self.head_size,
                self.intermediate_size,
                dropout_rate=self.dropout_rate,
                kernel_init_scale=self.kernel_init_scale,
                dtype=self.dtype,
            )(hidden_states, attention_mask, deterministic=deterministic)
            return attention

    @parallelize(memory_budget_per_device=160 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            rngs = {"dropout": batch['rng']}
            out = apply_fn(params, batch['hidden_states'],
                           batch['attention_mask'], deterministic,
                           rngs=rngs)
            return jnp.mean((out - batch['label']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    @parallelize(memory_budget_per_device=80 * (1 << 20),
                 devices=devices)
    def forward_step(optimizer, batch, apply_fn):
        rngs = {"dropout": batch['rng']}
        out = apply_fn(optimizer.target, batch['hidden_states'],
                       batch['attention_mask'], deterministic,
                       rngs=rngs)
        return out

    batch_size = 4
    seq_len = 128
    hidden_dim = 2048
    intermediate_size = hidden_dim * 4
    num_heads = 16
    per_head = hidden_dim // num_heads
    dropout_rate = 0.0
    deterministic = False

    hidden_states = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    model = Model(num_heads=num_heads, head_size=hidden_dim,
                  intermediate_size=intermediate_size, dropout_rate=dropout_rate)
    rngkey = jax.random.PRNGKey(0)

    params = model.init(rngkey, hidden_states, attention_mask, deterministic)
    #print(jax.tree_map(lambda x: x.shape, params))

    params = FrozenDict({
        "params": {
            "FlaxBertLayer_0": {
                "attention": {
                    "self": {
                        "query": {
                            "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                            "bias": jnp.ones((num_heads, per_head)),
                        },
                        "key": {
                            "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                            "bias": jnp.ones((num_heads, per_head)),
                        },
                        "value": {
                            "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                            "bias": jnp.ones((num_heads, per_head)),
                        },
                        "out": {
                            "kernel": jnp.ones((num_heads, per_head, hidden_dim)),
                            "bias": jnp.ones((hidden_dim,)),
                        },
                    },
                    "layer_norm": {
                        "beta": jnp.ones((hidden_dim,)),
                        "gamma": jnp.ones((hidden_dim,)),
                    },
                },
                "intermediate": {
                    "dense": {
                        "kernel": jnp.ones((hidden_dim, intermediate_size)),
                        "bias": jnp.ones((intermediate_size,))
                    }
                },
                "output": {
                    "dense": {
                        "kernel": jnp.ones((intermediate_size, hidden_dim)),
                        "bias": jnp.ones((hidden_dim,))
                    },
                    "layer_norm": {
                        "beta": jnp.ones((hidden_dim,)),
                        "gamma": jnp.ones((hidden_dim,)),
                    },
                },
            },
        },
    })

    optimizer = optim.GradientDescent(1e-2).create(params)

    if mode == 'forward':
        optimizer = forward_step(optimizer,
                                 {"hidden_states": hidden_states,
                                  "attention_mask": attention_mask,
                                  "label": label,
                                  "rng": rngkey},
                                 model.apply)
    else:
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               model.apply)

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    assert hlo_ir.count("channel_id") == 3
    assert hlo_ir.count("all-reduce(") == 3

    # all weight tensors should be split over the head dimension
    for name in ["query", "key", "value"]:
        weight_q = optimizer.target["params"]["FlaxBertLayer_0"]["attention"]["self"][name]["kernel"]
        assert weight_q.sharding_spec == pxla.ShardingSpec(
            sharding=(Chunked([1]), Chunked([4]), Chunked([1])),
            mesh_mapping=(ShardedAxis(0), ShardedAxis(1), ShardedAxis(2)),
        )
    weight0 = optimizer.target["params"]["FlaxBertLayer_0"]["intermediate"]["dense"]["kernel"]
    weight1 = optimizer.target["params"]["FlaxBertLayer_0"]["output"]["dense"]["kernel"]
    # column partitioned
    assert weight0.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([1]), Chunked([4])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
    )
    # row partitioned
    assert weight1.sharding_spec == pxla.ShardingSpec(
        sharding=(Chunked([4]), Chunked([1])),
        mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
    )


def test_n_bert_layer(mode='forward'):
    assert len(jax.devices()) >= 4
    devices = tuple(jax.devices()[:4])

    class Model(nn.Module):
        num_layers: int
        num_heads: int
        head_size: int
        intermediate_size: int
        dropout_rate: float = 0.0
        kernel_init_scale: float = 0.2
        dtype: jnp.dtype = jnp.float32

        @nn.compact
        def __call__(self, hidden_states, attention_mask, deterministic: bool=True):
            for i in range(self.num_layers):
                hidden_states = FlaxBertLayer(
                    self.num_heads,
                    self.head_size,
                    self.intermediate_size,
                    dropout_rate=self.dropout_rate,
                    kernel_init_scale=self.kernel_init_scale,
                    dtype=self.dtype,
                )(hidden_states, attention_mask, deterministic=deterministic)
            return hidden_states

    @parallelize(memory_budget_per_device=440 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            rngs = {"dropout": batch['rng']}
            out = apply_fn(params, batch['hidden_states'],
                           batch['attention_mask'], deterministic,
                           rngs=rngs)
            return jnp.mean((out - batch['label']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    @parallelize(memory_budget_per_device=80 * (1 << 20),
                 devices=devices)
    def forward_step(optimizer, batch, apply_fn):
        rngs = {"dropout": batch['rng']}
        out = apply_fn(optimizer.target, batch['hidden_states'],
                       batch['attention_mask'], deterministic,
                       rngs=rngs)
        return out

    batch_size = 4
    seq_len = 128
    num_layers = 2
    hidden_dim = 2304
    intermediate_size = hidden_dim * 4
    num_heads = 24
    per_head = hidden_dim // num_heads
    dropout_rate = 0.0
    deterministic = False

    hidden_states = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    model = Model(num_layers=num_layers,
                  num_heads=num_heads, head_size=hidden_dim,
                  intermediate_size=intermediate_size, dropout_rate=dropout_rate)
    rngkey = jax.random.PRNGKey(0)

    params = model.init(rngkey, hidden_states, attention_mask, deterministic)

    params = dict({
        "params": {}
    })

    for i in range(num_layers):
        params['params'][f"FlaxBertLayer_{i}"] = {
            "attention": {
                "self": {
                    "query": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "key": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "value": {
                        "kernel": jnp.ones((hidden_dim, num_heads, per_head)),
                        "bias": jnp.ones((num_heads, per_head)),
                    },
                    "out": {
                        "kernel": jnp.ones((num_heads, per_head, hidden_dim)),
                        "bias": jnp.ones((hidden_dim,)),
                    },
                },
                "layer_norm": {
                    "beta": jnp.ones((hidden_dim,)),
                    "gamma": jnp.ones((hidden_dim,)),
                },
            },
            "intermediate": {
                "dense": {
                    "kernel": jnp.ones((hidden_dim, intermediate_size)),
                    "bias": jnp.ones((intermediate_size,))
                }
            },
            "output": {
                "dense": {
                    "kernel": jnp.ones((intermediate_size, hidden_dim)),
                    "bias": jnp.ones((hidden_dim,))
                },
                "layer_norm": {
                    "beta": jnp.ones((hidden_dim,)),
                    "gamma": jnp.ones((hidden_dim,)),
                },
            },
        }

    optimizer = optim.GradientDescent(1e-2).create(params)

    if mode == 'forward':
        optimizer = forward_step(optimizer,
                                 {"hidden_states": hidden_states,
                                  "attention_mask": attention_mask,
                                  "label": label,
                                  "rng": rngkey},
                                 model.apply)
    else:
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                               model.apply)

    hlo_module = testing.last_compiled_executable().hlo_modules()[0]
    hlo_ir = hlo_module.to_string()

    print(hlo_ir)
    print(hlo_ir.count("channel_id"))
    print(hlo_ir.count("all-reduce("))

    #assert hlo_ir.count("channel_id") == 3
    #assert hlo_ir.count("all-reduce(") == 3

    ## all weight tensors should be split over the head dimension
    for i in range(num_layers):
        layer_name = f"FlaxBertLayer_{i}"
        for name in ["query", "key", "value"]:
            weight_q = optimizer.target["params"][layer_name]["attention"]\
                                       ["self"][name]["kernel"]
            print(layer_name, name, weight_q.sharding_spec)
            #assert weight_q.sharding_spec == pxla.ShardingSpec(
            #    sharding=(Chunked([1]), Chunked([4]), Chunked([1])),
            #    mesh_mapping=(ShardedAxis(0), ShardedAxis(1), ShardedAxis(2)),
            #)
        weight0 = optimizer.target["params"][layer_name]["attention"]["self"]["out"]["kernel"]
        weight1 = optimizer.target["params"][layer_name]["intermediate"]["dense"]["kernel"]
        weight2 = optimizer.target["params"][layer_name]["output"]["dense"]["kernel"]
        print(layer_name, "w_att_out", weight0.sharding_spec)
        print(layer_name, "w_inter", weight1.sharding_spec)
        print(layer_name, "w_outerput", weight2.sharding_spec)
        ## column partitioned
        #assert weight0.sharding_spec == pxla.ShardingSpec(
        #    sharding=(Chunked([1]), Chunked([4])),
        #    mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
        #)
        ## row partitioned
        #assert weight1.sharding_spec == pxla.ShardingSpec(
        #    sharding=(Chunked([4]), Chunked([1])),
        #    mesh_mapping=(ShardedAxis(0), ShardedAxis(1)),
        #)


if __name__ == "__main__":
    global_config.set_shard_parallel_strategy('auto_sharding')

    test_donate_buffer()
    test_2_layer_mlp()
    test_n_layer_mlp()
    test_attention(mode='train')
    test_bert_layer(mode='train')

    #test_n_bert_layer(mode='train')

