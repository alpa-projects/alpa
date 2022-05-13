from functools import partial
from typing import Callable, Optional, Tuple
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
from flax import optim

from alpa.model.bert_model import BertConfig
import flax.linen as nn
from alpa.model.model_util import (FlaxBaseModelOutput,
                                   FlaxMaskedLMOutput)

ACT2FN = {
    "gelu": partial(nn.gelu, approximate=False),
    "relu": nn.relu,
    "silu": nn.swish,
    "swish": nn.swish,
    "gelu_new": partial(nn.gelu, approximate=True),
}

class FlaxBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):

        if self.config.gradient_checkpointing:
            trans_func = partial(nn.remat, concrete=True)
        else:
            trans_func = lambda x: x

        self.word_embeddings = trans_func(nn.Embed)(
            self.config.vocab_size,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            dtype=self.dtype,
        )
        self.position_embeddings = trans_func(nn.Embed)(
            self.config.max_position_embeddings,
            self.config.hidden_size,
            embedding_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range),
            dtype=self.dtype,
        )

        if self.config.type_vocab_size > 0:
            self.token_type_embeddings = trans_func(nn.Embed)(
                self.config.type_vocab_size,
                self.config.hidden_size,
                embedding_init=jax.nn.initializers.normal(
                    stddev=self.config.initializer_range),
                dtype=self.dtype,
            )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                      dtype=self.dtype)

    def __call__(self,
                 input_ids,
                 token_type_ids,
                 position_ids,
                 attention_mask,
                 deterministic: bool = True):
        # Embed
        inputs_embeds = self.word_embeddings(input_ids.astype("i4"))
        position_embeds = self.position_embeddings(position_ids.astype("i4"))

        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(
                token_type_ids.astype("i4"))
        else:
            token_type_embeddings = 0.0

        # Sum all embeddings
        hidden_states = inputs_embeds + position_embeds + token_type_embeddings
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FlaxBertSelfAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.hidden_size % self.config.num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size`: {self.config.hidden_size} has to be a multiple of `num_attention_heads`: {self.config.num_attention_heads}"
            )

        self.qvk_combined = nn.Dense(
            self.config.hidden_size * 3,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
        )

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic=True,
                 output_attentions: bool = False):
        head_dim = self.config.hidden_size // self.config.num_attention_heads

        qvk_combined_states = self.qvk_combined(hidden_states)
        qvk_combined_states = qvk_combined_states.reshape(
            qvk_combined_states.shape[:2] + (-1, 3))
        query_states, value_states, key_states = jnp.split(qvk_combined_states,
                                                           3,
                                                           axis=3)

        query_states = query_states.reshape(hidden_states.shape[:2] +
                                            (self.config.num_attention_heads,
                                             head_dim))
        value_states = value_states.reshape(hidden_states.shape[:2] +
                                            (self.config.num_attention_heads,
                                             head_dim))
        key_states = key_states.reshape(hidden_states.shape[:2] +
                                        (self.config.num_attention_heads,
                                         head_dim))

        # Convert the boolean attention mask to an attention bias.
        if attention_mask is not None:
            # attention mask in the form of attention bias
            attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
            attention_bias = lax.select(
                attention_mask > 0,
                jnp.full(attention_mask.shape, 0.0).astype(self.dtype),
                jnp.full(attention_mask.shape, -1e10).astype(self.dtype),
            )
        else:
            attention_bias = None

        dropout_rng = None

        if not deterministic and self.config.attention_probs_dropout_prob > 0.0:
            dropout_rng = self.make_rng("dropout")

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
            dropout_rng=dropout_rng,
            dropout_rate=self.config.attention_probs_dropout_prob,
            broadcast_dropout=True,
            deterministic=deterministic,
            dtype=self.dtype,
            precision=None,
        )

        attn_output = jnp.einsum("...hqk,...khd->...qhd", attn_weights,
                                 value_states)
        attn_output = attn_output.reshape(attn_output.shape[:2] + (-1,))

        outputs = (attn_output,
                   attn_weights) if output_attentions else (attn_output,)
        return outputs


class FlaxBertSelfOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                      dtype=self.dtype)

    def __call__(self, hidden_states, input_tensor, deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class FlaxBertAttention(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.self = FlaxBertSelfAttention(self.config, dtype=self.dtype)
        self.output = FlaxBertSelfOutput(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic=True,
                 output_attentions: bool = False):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        attn_outputs = self.self(hidden_states,
                                 attention_mask,
                                 deterministic=deterministic,
                                 output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.output(attn_output,
                                    hidden_states,
                                    deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class FlaxBertIntermediate(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.intermediate_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.hidden_act]

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class FlaxBertOutput(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            dtype=self.dtype,
        )
        self.dropout = nn.Dropout(rate=self.config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                      dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_output,
                 deterministic: bool = True):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, deterministic=deterministic)
        hidden_states = self.LayerNorm(hidden_states + attention_output)
        return hidden_states


class FlaxBertLayer(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.attention = FlaxBertAttention(self.config, dtype=self.dtype)
        self.intermediate = FlaxBertIntermediate(self.config, dtype=self.dtype)
        self.output = FlaxBertOutput(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic: bool = True,
                 output_attentions: bool = False):

        if not isinstance(deterministic, bool):
            # A temporary hack to walkaround the bug in flax.nn.remat
            # Using `nn.remat(concrete=True)` works for regular use cases
            # (e.g., train_step, init) but does not work for init_dummy.
            # So we still need this hack.
            deterministic = True
            output_attentions = True

        attention_outputs = self.attention(hidden_states,
                                           attention_mask,
                                           deterministic=deterministic,
                                           output_attentions=output_attentions)
        attention_output = attention_outputs[0]

        hidden_states = self.intermediate(attention_output)
        hidden_states = self.output(hidden_states,
                                    attention_output,
                                    deterministic=deterministic)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class FlaxBertLayerCollection(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.gradient_checkpointing:
            trans_func = partial(nn.remat, concrete=True)
        else:
            trans_func = lambda x: x

        # Mixed rematerialization
        #layers = []
        #for i in range(self.config.num_hidden_layers):
        #    if i % 2 == 0:
        #        layer = trans_func(FlaxBertLayer)(self.config,
        #                                  name=str(i),
        #                                  dtype=self.dtype)
        #    else:
        #        layer = FlaxBertLayer(self.config,
        #                              name=str(i),
        #                              dtype=self.dtype)
        #    layers.append(layer)
        #self.layers = layers

        self.layers = [
            trans_func(FlaxBertLayer)(self.config,
                                      name=str(i),
                                      dtype=self.dtype)
            for i in range(self.config.num_hidden_layers)
        ]

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        id = 0
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(hidden_states,
                                  attention_mask,
                                  deterministic=deterministic,
                                  output_attentions=output_attentions)
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(last_hidden_state=hidden_states,
                                   hidden_states=all_hidden_states,
                                   attentions=all_attentions)


class FlaxBertEncoder(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = FlaxBertLayerCollection(self.config, dtype=self.dtype)

    def __call__(
        self,
        hidden_states,
        attention_mask,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        return self.layer(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class FlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(self.config, dtype=self.dtype)

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
        hidden_states = self.embeddings(input_ids,
                                        token_type_ids,
                                        position_ids,
                                        attention_mask,
                                        deterministic=deterministic)
        outputs = self.encoder(
            hidden_states,
            attention_mask,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        if not return_dict:
            # if pooled is None, don't return it
            return (hidden_states,) + outputs[1:]

        return FlaxBaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OPTForLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = FlaxBertModule(config=self.config,
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
