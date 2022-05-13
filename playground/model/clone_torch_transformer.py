from functools import partial
from typing import Callable, Optional, Tuple
import numpy as np

import jax
import jax.numpy as jnp
from flax import optim

from alpa.model.bert_model import BertConfig
from alpa.model.gpt_model import FlaxGPTForLMModule, FlaxBertModule, FlaxMaskedLMOutput
import flax.linen as nn


class OPTForLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = FlaxBertModule(config=self.config,
                                           add_pooling_layer=False,
                                           dtype=self.dtype)

        if self.config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = nn.Dense(self.config.vocab_size,
                                    dtype=self.dtype,
                                    use_bias=False)
        self.decoder_bias = self.param("bias", self.bias_init,
                                       (self.config.vocab_size,))

    def __call__(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # Model
        outputs = self.transformers(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        if self.config.tie_word_embeddings:
            if self.dtype == jnp.float16:
                shared_embedding = self.transformers.embeddings.word_embeddings.embedding_fp16
            else:
                shared_embedding = self.transformers.variables["params"][
                    "embeddings"]["word_embeddings"]["embedding"]
            assert self.decoder is None
            logits = hidden_states @ shared_embedding.T
        else:
            assert self.decoder is not None
            logits = self.decoder(hidden_states)

        logits += jnp.asarray(self.decoder_bias, self.dtype)

        # Compute the prediction scores
        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def test_gpt_lm():
    batch_size = 1
    seq_len = 9
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
    input_ids = jnp.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=jnp.int32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    print("input_ids", input_ids.shape, input_ids)

    model = OPTForLMModule(
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

if __name__ == "__main__":
    test_gpt_lm()
