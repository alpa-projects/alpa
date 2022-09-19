# flake8: noqa
"""Model definition of BERT.
Copied from https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_flax_bert.py"""
from functools import partial
from typing import Callable

import numpy as np

from flax import linen as nn, optim
import jax
from jax import lax
import jax.numpy as jnp

from alpa.model.model_util import (FlaxBaseModelOutput,
                                   FlaxBaseModelOutputWithPooling,
                                   FlaxBertForPreTrainingOutput,
                                   FlaxMaskedLMOutput, TrainState)
from alpa.model.model_util import TrainState
from alpa.pipeline_parallel.primitive_def import mark_pipeline_boundary


class BertConfig:

    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 gradient_checkpointing=False,
                 position_embedding_type="absolute",
                 use_cache=True,
                 tie_word_embeddings=True,
                 add_manual_pipeline_markers=False,
                 pipeline_mp_size=0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.add_manual_pipeline_markers = add_manual_pipeline_markers
        self.pipeline_mp_size = pipeline_mp_size


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
            broadcast_dropout=False,
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

            if self.config.add_manual_pipeline_markers:
                layers_per_stage = self.config.num_hidden_layers // self.config.pipeline_mp_size
                assert self.config.num_hidden_layers % self.config.pipeline_mp_size == 0
                if i % layers_per_stage == layers_per_stage - 1 and i != len(
                        self.layers) - 1:
                    mark_pipeline_boundary()

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


class FlaxBertPooler(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.dense = nn.Dense(
            self.config.hidden_size,
            kernel_init=jax.nn.initializers.normal(
                self.config.initializer_range),
            dtype=self.dtype,
        )

    def __call__(self, hidden_states):
        cls_hidden_state = hidden_states[:, 0]
        cls_hidden_state = self.dense(cls_hidden_state)
        return nn.tanh(cls_hidden_state)


class FlaxBertPredictionHeadTransform(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.dense = nn.Dense(self.config.hidden_size, dtype=self.dtype)
        self.activation = ACT2FN[self.config.hidden_act]
        self.LayerNorm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                      dtype=self.dtype)

    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        return self.LayerNorm(hidden_states)


class FlaxBertLMPredictionHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transform = FlaxBertPredictionHeadTransform(self.config,
                                                         dtype=self.dtype)
        if self.config.tie_word_embeddings:
            self.decoder = None
        else:
            self.decoder = nn.Dense(self.config.vocab_size,
                                    dtype=self.dtype,
                                    use_bias=False)
        self.bias = self.param("bias", self.bias_init,
                               (self.config.vocab_size,))

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.transform(hidden_states)

        if shared_embedding is not None:
            assert self.decoder is None
            hidden_states = hidden_states @ shared_embedding.T
        else:
            assert self.decoder is not None
            hidden_states = self.decoder(hidden_states)

        hidden_states += self.bias
        return hidden_states


class FlaxBertOnlyMLMHead(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config,
                                                    dtype=self.dtype)

    def __call__(self, hidden_states, shared_embedding=None):
        hidden_states = self.predictions(hidden_states,
                                         shared_embedding=shared_embedding)
        return hidden_states


class FlaxBertOnlyNSPHead(nn.Module):
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, pooled_output):
        return self.seq_relationship(pooled_output)


class FlaxBertPreTrainingHeads(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.predictions = FlaxBertLMPredictionHead(self.config,
                                                    dtype=self.dtype)
        self.seq_relationship = nn.Dense(2, dtype=self.dtype)

    def __call__(self, hidden_states, pooled_output, shared_embedding=None):
        prediction_scores = self.predictions(hidden_states,
                                             shared_embedding=shared_embedding)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class FlaxBertModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True

    def setup(self):
        self.embeddings = FlaxBertEmbeddings(self.config, dtype=self.dtype)
        self.encoder = FlaxBertEncoder(self.config, dtype=self.dtype)
        if self.add_pooling_layer:
            self.pooler = FlaxBertPooler(self.config, dtype=self.dtype)

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
        pooled = self.pooler(hidden_states) if self.add_pooling_layer else None

        if not return_dict:
            # if pooled is None, don't return it
            if pooled is None:
                return (hidden_states,) + outputs[1:]
            return (hidden_states, pooled) + outputs[1:]

        return FlaxBaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxBertForPreTrainingModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config, dtype=self.dtype)
        self.cls = FlaxBertPreTrainingHeads(config=self.config,
                                            dtype=self.dtype)

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
        outputs = self.bert(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.tie_word_embeddings:
            shared_embedding = self.bert.variables["params"]["embeddings"][
                "word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        hidden_states = outputs[0]
        pooled_output = outputs[1]

        prediction_scores, seq_relationship_score = self.cls(
            hidden_states, pooled_output, shared_embedding=shared_embedding)

        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]

        return FlaxBertForPreTrainingOutput(
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class FlaxBertForMaskedLMModule(nn.Module):
    config: BertConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.bert = FlaxBertModule(config=self.config,
                                   add_pooling_layer=False,
                                   dtype=self.dtype)
        self.cls = FlaxBertOnlyMLMHead(config=self.config, dtype=self.dtype)

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
        outputs = self.bert(
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
            shared_embedding = self.bert.variables["params"]["embeddings"][
                "word_embeddings"]["embedding"]
        else:
            shared_embedding = None

        # Compute the prediction scores
        logits = self.cls(hidden_states, shared_embedding=shared_embedding)

        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def test_bert_layer():
    batch_size = 64
    seq_len = 64
    hidden_size = 768

    hidden_states = jnp.ones((batch_size, seq_len, hidden_size),
                             dtype=jnp.float32)
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    label = jnp.ones((batch_size, seq_len, hidden_size), dtype=jnp.float32)

    # Init model and optimizer
    model = FlaxBertLayer(BertConfig(hidden_size=hidden_size))
    rngkey = jax.random.PRNGKey(0)
    params = model.init(rngkey, hidden_states, attention_mask)
    optimizer = optim.GradientDescent(1e-2).create(params)

    def train_step(optimizer, batch):

        def loss_func(params):
            rngs = {"dropout": batch["rng"]}
            out = model.apply(params,
                              batch["hidden_states"],
                              batch["attention_mask"],
                              rngs=rngs)[0]
            return jnp.mean((out - batch["label"])**2)

        grad = jax.grad(loss_func)(optimizer.target)
        new_optimizer = optimizer.apply_gradient(grad)
        return new_optimizer

    # JIT compile
    #optimizer = train_step(optimizer,
    #                       {"hidden_states": hidden_states,
    #                        "attention_mask": attention_mask,
    #                        "label": label,
    #                        "rng": rngkey})

    jaxpr = jax.make_jaxpr(train_step)(optimizer, {
        "hidden_states": hidden_states,
        "attention_mask": attention_mask,
        "label": label,
        "rng": rngkey
    })
    print(jaxpr)


def test_bert_mlm():
    batch_size = 64
    seq_len = 64
    hidden_size = 128
    num_attention_heads = 4
    num_hidden_layers = 2
    vocab_size = 1024

    @partial(jax.jit, static_argnums=(2,))
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
    token_type_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    model = FlaxBertForMaskedLMModule(
        BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            num_hidden_layers=num_hidden_layers,
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
    #test_bert_layer()
    test_bert_mlm()
