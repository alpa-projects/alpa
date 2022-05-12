from functools import partial

import jax
import jax.numpy as jnp
from flax import optim

from alpa.model.bert_model import BertConfig
from alpa.model.gpt_model import FlaxGPTForLMModule


def test_gpt_lm():
    batch_size = 64
    seq_len = 64
    hidden_size = 128
    num_attention_heads = 4
    num_hidden_layers = 2
    vocab_size = 1024

    # @partial(jax.jit, static_argnums=(2,))
    def train_step(optimizer, batch, apply_func):

        def loss_func(params):
            rngs = {"dropout": batch["rng"]}
            logits = apply_func(params,
                                batch["input_ids"],
                                batch["attention_mask"],
                                batch["token_type_ids"],
                                batch["position_ids"],
                                rngs=rngs)[0]
            label_mask = jnp.where(batch["labels"] > 0, 1.0, 0.0)
            labels = jax.nn.one_hot(batch["labels"], logits.shape[-1])
            loss = -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1),
                            axis=-1)
            loss = (label_mask * loss).sum() / label_mask.sum()
            return loss

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # Init model and optimizer
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    print("input_ids", input_ids.shape, input_ids)

    model = FlaxGPTForLMModule(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=0,
        ))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, input_ids, attention_mask, token_type_ids,
                        position_ids)
    optimizer = optim.GradientDescent(1e-2).create(params)

    # JIT compile
    train_step(
        optimizer, {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "labels": labels,
            "rng": rngkey
        }, model.apply)
