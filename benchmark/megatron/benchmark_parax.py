from functools import partial
import os
import unittest

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim
from transformers.models.bert.modeling_flax_bert import FlaxBertAttention, FlaxBertLayer

from parax import parallelize, global_config, testing

from timeit import timeit

MB = 1024 ** 2


def replicate(a, devices):
    a = jax.pmap(lambda x, y: x, in_axes=(None, 0), out_axes=None, devices=devices)\
                (a, jnp.ones(len(devices)))
    return a


def block_until_ready(x):
    jax.tree_util.tree_leaves(x)[-1].block_until_ready()


def benchmark_2_layer_mlp():
    assert len(jax.local_devices()) >= 4
    devices = tuple(jax.local_devices()[:4])
    global_config.set_shard_parallel_strategy('auto_sharding')

    class Model(nn.Module):
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim * 4)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.hidden_dim)(x)
            return x

    @parallelize(memory_budget_per_device=300 * (1 << 20),
                 devices=devices)
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch['x'])
            return jnp.mean((out - batch['y']) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    batch_size = 4
    seq_len = 512
    hidden_dim = 2304

    # Prepare input
    x = jnp.ones((batch_size, seq_len, hidden_dim))
    y = jnp.ones((batch_size, seq_len, hidden_dim))
    x = replicate(x, devices)
    y = replicate(y, devices)

    # Initialize model
    model = Model(hidden_dim=hidden_dim)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, x)
    optimizer = optim.GradientDescent(1e-2).create(params)
    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]
        block_until_ready(optimizer)
        optimizer = train_step(optimizer,
                               {"x": x, "y": y},
                               model.apply)
        block_until_ready(optimizer)
        closure[0] = optimizer


    # Benchmark time cost
    func()
    func()
    stmt = "func()"
    number = 100
    cost = timeit(stmt, globals={**globals(), **locals()},
                  number=number) / number

    # Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    print(f"#communication: {hlo_ir.count('channel_id')}")
    print(f"#all-reduce: {hlo_ir.count('all-reduce(')}")

    print(f"Time: {cost * 1e3:.2f} ms")



def benchmark_bert_layer():
    assert len(jax.local_devices()) >= 4
    devices = tuple(jax.local_devices()[:4])
    global_config.set_shard_parallel_strategy('auto_sharding')

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

    @parallelize(memory_budget_per_device=800 * (1 << 20),
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

    batch_size = 4
    seq_len = 512
    hidden_dim = 2304
    intermediate_size = hidden_dim * 4
    num_heads = 24
    per_head = hidden_dim // num_heads

    dropout_rate = 0.0
    deterministic = False

    # Prepare input
    hidden_states = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_dim), dtype=jnp.float32)

    hidden_states = replicate(hidden_states, devices)
    attention_mask = replicate(attention_mask, devices)
    label = replicate(label, devices)

    # Initialize model
    model = Model(num_heads=num_heads, head_size=hidden_dim,
                  intermediate_size=intermediate_size, dropout_rate=dropout_rate)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask, deterministic)

    optimizer = optim.GradientDescent(1e-2).create(params)

    # Define benchmark function
    closure = [optimizer]
    def func():
        optimizer = closure[0]
        block_until_ready(optimizer)
        optimizer = train_step(optimizer,
                               {"hidden_states": hidden_states,
                                "attention_mask": attention_mask,
                                "label": label,
                                "rng": rngkey},
                                model.apply)
        block_until_ready(optimizer)
        closure[0] = optimizer

    # Benchmark time cost
    func()
    func()
    stmt = "func()"
    number = 100
    cost = timeit(stmt, globals={**globals(), **locals()},
                  number=number) / number

    # Check sharding strategy
    hlo_module = testing.last_compiled_executable.hlo_modules()[0]
    hlo_ir = hlo_module.to_string()
    print(f"#communication: {hlo_ir.count('channel_id')}")
    print(f"#all-reduce: {hlo_ir.count('all-reduce(')}")

    print(f"Time: {cost * 1e3:.2f} ms")



if __name__ == '__main__':
    #benchmark_2_layer_mlp()
    benchmark_bert_layer()

