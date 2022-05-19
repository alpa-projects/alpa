import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple
import math
import numpy as np

import jax
from jax import lax
import jax.numpy as jnp
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

@dataclass
class OPTConfig:
    # Inherited from OPT
    no_progress_bar = False
    log_interval = 10
    log_format = 'json'
    log_file = None
    wandb_project = None
    azureml_logging = False
    seed = 1
    cpu = False
    tpu = False
    bf16 = False
    memory_efficient_bf16 = False
    fp16 = True
    memory_efficient_fp16 = True
    fp16_no_flatten_grads = False
    fp16_init_scale = 4
    fp16_scale_window = None
    fp16_scale_tolerance = 0.0
    min_loss_scale = 0.0001
    threshold_loss_scale = None
    user_dir = None
    empty_cache_freq = 0
    all_gather_list_size = 16384
    model_parallel_size = 2
    quantization_config_path = None
    profile = False
    reset_logging = False
    suppress_crashes = False
    use_plasma_view = False
    plasma_path = '/tmp/plasma'
    log_nvidia_smi = False
    use_tutel_moe = False
    new_profiler = False
    criterion = 'vocab_parallel_cross_entropy'
    tokenizer = None
    bpe = None
    simul_type = None
    optimizer = 'adam'
    lr_scheduler = 'polynomial_decay'
    scoring = 'bleu'
    task = 'streaming_language_modeling'
    num_workers = 8
    num_workers_valid = 1
    skip_invalid_size_inputs_valid_test = False
    max_tokens = None
    batch_size = 1
    required_batch_size_multiple = 1
    required_seq_len_multiple = 1
    dataset_impl = None
    data_buffer_size = 10
    validate_interval = 1
    validate_interval_updates = 1000
    validate_after_updates = 0
    fixed_validation_seed = None
    disable_validation = False
    max_tokens_valid = None
    batch_size_valid = 1
    max_valid_steps = None
    curriculum = 0
    gen_subset = 'test'
    num_shards = 1
    shard_id = 0
    distributed_world_size = 32
    distributed_rank = 0
    distributed_backend = 'nccl'
    distributed_init_method = None
    distributed_port = 10079
    device_id = 0
    distributed_no_spawn = False
    ddp_backend = 'fully_sharded'
    bucket_cap_mb = 25
    fix_batches_to_gpus = False
    find_unused_parameters = False
    fast_stat_sync = False
    heartbeat_timeout = -1
    broadcast_buffers = False
    slowmo_momentum = None
    slowmo_algorithm = 'LocalSGD'
    localsgd_frequency = 3
    nprocs_per_node = 8
    pipeline_model_parallel = False
    pipeline_balance = None
    pipeline_devices = None
    pipeline_chunks = 0
    pipeline_encoder_balance = None
    pipeline_encoder_devices = None
    pipeline_decoder_balance = None
    pipeline_decoder_devices = None
    pipeline_checkpoint = 'never'
    zero_sharding = 'none'
    no_reshard_after_forward = False
    fp32_reduce_scatter = False
    cpu_offload = False
    use_sharded_state = True
    gradient_predivide_factor = None
    arch = 'transformer_lm_megatron'
    max_epoch = 0
    max_update = 572204
    stop_time_hours = 0
    clip_norm = 1.0
    clip_norm_type = 'l2'
    skip_gradient_update_on_clip_norm = False
    sentence_avg = False
    update_freq = [1]
    lr = [0.0006]
    stop_min_lr = -1.0
    use_bmuf = False
    train_with_epoch_remainder_batch = False
    finetune_from_model = None
    reset_dataloader = False
    reset_lr_scheduler = False
    reset_meters = False
    reset_optimizer = False
    optimizer_overrides = '{}'
    save_interval = 1
    save_interval_updates = 1000
    keep_interval_updates = -1
    keep_last_epochs = -1
    keep_best_checkpoints = -1
    no_save = False,
    no_epoch_checkpoints = True
    no_last_checkpoints = False
    no_best_checkpoints = True
    no_save_optimizer_state = False
    no_save_optimizer_state_on_training_finished = False
    best_checkpoint_metric = 'loss'
    maximize_best_checkpoint_metric = False
    patience = -1
    checkpoint_suffix = ''
    checkpoint_shard_count = 1
    load_checkpoint_on_all_dp_ranks = False
    write_checkpoints_asynchronously = True
    data = '/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp29'
    vocab_filename = '/home/ubuntu/efs/metaseq/run_model/vocab.json'
    merges_filename = '/home/ubuntu/efs/metaseq/run_model/merges.txt'
    end_of_document_symbol = '</s>'
    sample_break_mode = 'none'
    tokens_per_sample = 2048
    max_source_positions = None
    max_target_positions = 2048
    adam_betas = '(0.9, 0.95)'
    adam_eps = 1e-08
    weight_decay = 0.1
    use_old_adam = False
    fp16_adam_stats = False
    block_wise = False
    warmup_updates = 715
    force_anneal = None
    end_learning_rate = 5.9999999999999995e-05
    zero_lr_warmup_steps = 0
    power = 1.0
    total_num_update = '572204'
    pad = 1
    eos = 2
    unk = 3
    distribute_checkpointed_activations = True
    full_megatron_init = True
    megatron_init_sigma = 0.006
    activation_fn = 'relu'
    share_decoder_input_output_embed = True
    decoder_layers = 12
    decoder_embed_dim = 768
    decoder_ffn_embed_dim = 3072
    decoder_attention_heads = 12
    decoder_learned_pos = True
    no_scale_embedding = True
    dropout = 0.1
    attention_dropout = 0.1
    no_emb_dropout = True
    no_seed_provided = False
    activation_dropout = 0.0
    relu_dropout = 0.0
    decoder_output_dim = 768
    decoder_input_dim = 768
    decoder_normalize_before = True
    no_decoder_final_norm = False
    adaptive_softmax_cutoff = None
    adaptive_softmax_dropout = 0
    adaptive_softmax_factor = 4
    no_token_positional_embeddings = False
    character_embeddings = False
    character_filters = '[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]'
    character_embedding_dim = 4
    char_embedder_highway_layers = 2
    adaptive_input = False
    adaptive_input_factor = 4
    adaptive_input_cutoff = None
    tie_adaptive_weights = False
    tie_adaptive_proj = False
    decoder_learned_sinusoidal = False
    decoder_layerdrop = 0.0
    decoder_layers_to_keep = None
    layernorm_embedding = False
    quant_noise_pq = 0.0
    quant_noise_pq_block_size = 8
    quant_noise_scalar = 0.0
    add_bos_token = False
    _name = 'transformer_lm'
    checkpoint_activations = False
    offload_activations = False
    min_params_to_wrap = 100000000
    use_stable_embedding = False
    scale_fc = False
    scale_attn = False
    scale_heads = False
    alibi = False
    fsdp_checkpoint_wrap_layer_frequency = 1
    tensor_parallel_init_model_on_gpu = False
    sync_ln_variance = False
    world_size = '${distributed_training.distributed_world_size}'
    # Added
    vocab_size = 50272
    layer_norm_eps = 0.00001


class OPTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        assert not self.config.use_stable_embedding
        self.embed_scale = 1.0 if self.config.no_scale_embedding else math.sqrt(
            self.config.decoder_embed_dim)
        self.word_embeddings = nn.Embed(
            self.config.vocab_size,
            self.config.decoder_input_dim,
            dtype=self.dtype,
        )
        assert self.config.max_target_positions is not None
        assert self.config.decoder_learned_pos
        self.position_embeddings = nn.Embed(
            self.config.max_target_positions + self.config.pad + 1,
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        )
        self.project_in_dim = nn.Dense(
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        ) if self.config.decoder_input_dim != self.config.decoder_embed_dim else None

    def __call__(self,
                 input_ids,
                 position_ids,
                 attention_mask,
                 deterministic: bool = True):
        # Embed
        inputs_embeds = self.embed_scale * self.word_embeddings(input_ids.astype("i4"))
        if self.project_in_dim is not None:
            inputs_embeds = self.project_in_dim(inputs_embeds)
        position_embeds = self.position_embeddings(position_ids.astype("i4"))

        # Sum all embeddings
        hidden_states = inputs_embeds + position_embeds
        return hidden_states


class FlaxBertSelfAttention(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        if self.config.decoder_embed_dim % self.config.decoder_attention_heads != 0:
            raise ValueError(
                f"`decoder_embed_dim`: {self.config.decoder_embed_dim} has to be a "
                f"multiple of `decoder_attention_heads`: {self.config.decoder_attention_heads}"
            )

        self.qvk_combined = nn.Dense(
            self.config.decoder_embed_dim * 3,
            dtype=self.dtype,
        )

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic=True,
                 output_attentions: bool = False):
        head_dim = self.config.decoder_embed_dim // self.config.decoder_attention_heads

        qvk_combined_states = self.qvk_combined(hidden_states)
        qvk_combined_states = qvk_combined_states.reshape(
            qvk_combined_states.shape[:2] + (-1, 3))
        query_states, value_states, key_states = jnp.split(qvk_combined_states,
                                                           3,
                                                           axis=3)

        query_states = query_states.reshape(hidden_states.shape[:2] +
                                            (self.config.decoder_attention_heads,
                                             head_dim))
        value_states = value_states.reshape(hidden_states.shape[:2] +
                                            (self.config.decoder_attention_heads,
                                             head_dim))
        key_states = key_states.reshape(hidden_states.shape[:2] +
                                        (self.config.decoder_attention_heads,
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

        attn_weights = nn.attention.dot_product_attention_weights(
            query_states,
            key_states,
            bias=attention_bias,
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


class OPTAttention(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        assert self.config.decoder_normalize_before
        self.self = FlaxBertSelfAttention(self.config, dtype=self.dtype)
        self.dense = nn.Dense(
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic=True,
                 output_attentions: bool = False):
        # Attention mask comes in as attention_mask.shape == (*batch_sizes, kv_length)
        # FLAX expects: attention_mask.shape == (*batch_sizes, 1, 1, kv_length) such that it is broadcastable
        # with attn_weights.shape == (*batch_sizes, num_heads, q_length, kv_length)
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        attn_outputs = self.self(hidden_states,
                                 attention_mask,
                                 deterministic=deterministic,
                                 output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        hidden_states = self.dense(attn_output)
        hidden_states = hidden_states + residual
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


class OPTFFN(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.fc1 = nn.Dense(
            self.config.decoder_ffn_embed_dim,
            dtype=self.dtype,
        )
        self.activation = ACT2FN[self.config.activation_fn]
        self.fc2 = nn.Dense(
            self.config.decoder_embed_dim,
            dtype=self.dtype,
        )
        self.layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps,
                                       dtype=self.dtype)

    def __call__(self, hidden_states):
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.activation(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


def load_params(params, path):
    def load_param(param_key, checkpoint_key):
        param_dict = params
        param_keys = param_key.split('.')
        loaded_array = np.load(os.path.join(path, checkpoint_key))
        for i, key in enumerate(param_keys):
            if i == len(param_keys) - 1:
                assert param_dict[key].shape == loaded_array.shape
                param_dict[key] = loaded_array
            else:
                param_dict = param_dict[key]
    load_param("params.transformers.embeddings.word_embeddings.embedding", "decoder.embed_tokens.weight")
    return params


def print_params(params, prefix=""):
    for key, value in params.items():
        if isinstance(value, dict):
            print_params(value, prefix=prefix + key + ".")
        else:
            print(prefix + key, value.shape)


class FlaxBertLayer(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        assert self.config.decoder_normalize_before
        assert not getattr(self.config, "cross_self_attention", False)
        assert not getattr(self.config, "scale_heads", False)
        assert not getattr(self.config, "scale_attn", False)
        assert not getattr(self.config, "scale_fc", False)
        self.attention = OPTAttention(self.config, dtype=self.dtype)
        self.ffn = OPTFFN(self.config, dtype=self.dtype)

    def __call__(self,
                 hidden_states,
                 attention_mask,
                 deterministic: bool = True,
                 output_attentions: bool = False):

        attention_outputs = self.attention(hidden_states,
                                           attention_mask,
                                           deterministic=deterministic,
                                           output_attentions=output_attentions)
        attention_output = attention_outputs[0]

        hidden_states = self.ffn(attention_output)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attention_outputs[1],)
        return outputs


class OPTTransformerLayerCollection(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layers = [
            FlaxBertLayer(self.config,
                          name=str(i),
                          dtype=self.dtype)
            for i in range(self.config.decoder_layers)
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

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        outputs = (hidden_states,)

        if not return_dict:
            return tuple(v for v in outputs if v is not None)

        return FlaxBaseModelOutput(last_hidden_state=hidden_states,
                                   hidden_states=all_hidden_states,
                                   attentions=all_attentions)


class OPTTransformerEncoder(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.layer = OPTTransformerLayerCollection(self.config, dtype=self.dtype)

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


class OPTTransformerModule(nn.Module):
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    def setup(self):
        self.embeddings = OPTEmbeddings(self.config, dtype=self.dtype)
        self.encoder = OPTTransformerEncoder(self.config, dtype=self.dtype)

    def __call__(
        self,
        input_ids,
        attention_mask,
        position_ids,
        deterministic: bool = True,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        hidden_states = self.embeddings(input_ids,
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
    config: OPTConfig
    dtype: jnp.dtype = jnp.float32
    bias_init: Callable[..., np.ndarray] = jax.nn.initializers.zeros

    def setup(self):
        self.transformers = OPTTransformerModule(config=self.config,
                                                 dtype=self.dtype)

        self.project_out_dim = nn.Dense(
            self.config.decoder_input_dim,
            dtype=self.dtype,
        ) if self.config.decoder_input_dim != self.config.decoder_embed_dim else None

        if self.config.share_decoder_input_output_embed:
            self.decoder = None
        else:
            self.decoder = nn.Dense(self.config.vocab_size,
                                    dtype=self.dtype,
                                    use_bias=False)

    def __call__(
        self,
        input_ids,
        attention_mask,
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
            position_ids,
            deterministic=deterministic,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]

        if self.project_out_dim is not None:
            hidden_states = self.project_out_dim(hidden_states)

        if self.config.share_decoder_input_output_embed:
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

        # Compute the prediction scores
        if not return_dict:
            return (logits,) + outputs[1:]

        return FlaxMaskedLMOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def test_gpt_lm():
    config = OPTConfig()
    batch_size = config.batch_size
    seq_len = 9

    # @partial(jax.jit, static_argnums=(2,))
    def inference_step(batch, apply_func):
        logits = apply_func(params,
                            batch["input_ids"],
                            batch["attention_mask"],
                            batch["position_ids"])[0]
        return logits

    # Init model and optimizer
    input_ids = jnp.array([[5625,   16,   10, 2721,  183,    8,   38,  236,    7]], dtype=jnp.int32)
    # TODO: attention_mask should be triu
    attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    # TODO: set up position_ids correctly
    position_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)

    print("input_ids", input_ids.shape, input_ids)

    model = OPTForLMModule(config)
    rngkey = jax.random.PRNGKey(0)

    params = model.init(rngkey, input_ids, attention_mask,
                        position_ids)
    print_params(params.unfreeze())
    params = load_params(params.unfreeze(), "numpy_weights")

    print("=" * 40 + " after init " + "=" * 40)
    # JIT compile
    inference_step({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, model.apply)

if __name__ == "__main__":
    test_gpt_lm()
