from flax import optim, linen as nn
import jax
from jax import numpy as jnp

import alpa
from alpa.model.bert_model import FlaxBertLayer, BertConfig


def inspect_params(optimizer):
    """For debug usage."""
    print(jax.tree_util.tree_map(lambda x: (x.shape, x.dtype), optimizer.target))


def test_mlp():
    batch_size = 16
    hidden_size = 128

    class Model(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(features=hidden_size, dtype=jnp.float16)(x)
            x = nn.relu(x)
            x = nn.Dense(features=hidden_size, dtype=jnp.float16)(x)
            return x

    @alpa.parallelize
    def train_step(optimizer, batch, apply_fn):
        def loss_func(params):
            out = apply_fn(params, batch["x"])
            return jnp.mean((out - batch["y"]) ** 2, dtype=jnp.float16) * 0.1234

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    x = jnp.ones((batch_size, hidden_size), dtype=jnp.float16)
    y = jnp.ones((batch_size, hidden_size), dtype=jnp.float16)

    # Init model and optimizer
    model = Model()
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, x)
    optimizer = optim.GradientDescent(1e-2).create(params)

    # JIT compile
    optimizer = train_step(optimizer, {"x": x, "y": y}, model.apply)


def test_bert_layer():
    batch_size = 64
    seq_len = 64
    hidden_size = 768

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float16)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float16)

    # Init model and optimizer
    model = FlaxBertLayer(BertConfig(
        hidden_size=hidden_size,
    ), dtype=jnp.float16)
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
    optimizer = optim.GradientDescent(1e-2).create(params)

    @alpa.parallelize
    def train_step(optimizer, batch):
        def loss_func(params):
            rngs = {"dropout": batch["rng"]}
            out = model.apply(params,
                              batch["hidden_states"],
                              batch["attention_mask"],
                              rngs=rngs)[0]
            return jnp.mean((out - batch["label"]) ** 2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # JIT compile
    optimizer = train_step(optimizer,
                           {"hidden_states": hidden_states,
                            "attention_mask": attention_mask,
                            "label": label,
                            "rng": rngkey})
    inspect_params(optimizer)


if __name__ == "__main__":
    #test_mlp()
    test_bert_layer()

