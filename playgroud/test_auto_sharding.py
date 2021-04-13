from functools import partial
import os

import numpy as np

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import optim
from flax.core.frozen_dict import FrozenDict, freeze
from paranum import parallelize

from transformers.models.bert.modeling_flax_bert import FlaxBertAttention


def test_donate_buffer():
    @parallelize(donate_argnums=(0,), memory_budget_per_device=5 * 1024 **2)
    def add_one(x):
        x = x + 1
        return x

    a = jnp.ones((1024, 1024))
    b = add_one(a)


def test_matmul():

    @parallelize
    def matmul(a, b):
        a =  a @ b
        return a

    x = jnp.ones((128, 128))
    y = jnp.ones((128, 128))

    c = matmul(x, y)
    c.block_until_ready()

    np.testing.assert_allclose(np.array(c), np.array(x @ y))


def test_mlp():
    class Model(nn.Module):
        hidden_dim: int

        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
            #x = nn.relu(x)
            x = nn.Dense(features=self.hidden_dim, use_bias=False)(x)
            return x

    @parallelize(memory_budget_per_device=50 * (1 << 20))
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

    model = Model(hidden_dim=hidden_dim)
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


def test_attention():
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

    @parallelize
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

    @parallelize
    def forward_step(optimizer, batch, apply_fn):
        rngs = {"dropout": batch['rng']}
        out = apply_fn(optimizer.target, batch['hidden_states'],
                       batch['attention_mask'], deterministic,
                       rngs=rngs)
        return out

    batch_size = 4
    seq_len = 512
    num_heads = 12
    hidden_dim = 768
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
                        "kernel": jnp.ones((hidden_dim, num_heads, hidden_dim // num_heads)),
                        "bias": jnp.ones((num_heads, hidden_dim // num_heads)),
                    },
                    "key": {
                        "kernel": jnp.ones((hidden_dim, num_heads, hidden_dim // num_heads)),
                        "bias": jnp.ones((num_heads, hidden_dim // num_heads)),
                    },
                    "value": {
                        "kernel": jnp.ones((hidden_dim, num_heads, hidden_dim // num_heads)),
                        "bias": jnp.ones((num_heads, hidden_dim // num_heads)),
                    },
                    "out": {
                        "kernel": jnp.ones((num_heads, hidden_dim // num_heads, hidden_dim)),
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
    optimizer = train_step(optimizer,
                          {"hidden_states": hidden_states,
                           "attention_mask": attention_mask,
                           "label": label,
                           "rng": rngkey
                           }, model.apply)

    #optimizer = forward_step(optimizer,
    #                         {"hidden_states": hidden_states,
    #                          "attention_mask": attention_mask,
    #                          "label": label}, model.apply)


if __name__ == "__main__":
    #test_donate_buffer()
    #test_matmul()
    test_mlp()
    #test_attention()

