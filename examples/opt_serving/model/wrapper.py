"""Wrap models to make them compatible with huggingface's generator API."""
from collections import defaultdict
import os
from typing import Sequence, Any

import alpa
from alpa.device_mesh import DistributedArray
from alpa.mesh_executable import get_index_select_mesh_executable
import jax
from jax import xla
from jax import ShapeDtypeStruct, ShapedArray
from jax.interpreters import pxla
from jax.interpreters.pxla import NoSharding, Replicated, ShardingSpec
import jax.numpy as jnp
import numpy as np
import torch
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass
from transformers import OPTForCausalLM, GPT2LMHeadModel

from opt_serving.model.opt_model import (get_opt_config,
                                         get_pipeshard_executable,
                                         load_multi_executable_params_dis_array,
                                         init_multi_executable_cache_dis_array,
                                         load_params_np, init_cache_np,
                                         get_jax_executable)
from opt_serving.model.opt_utils import (TransformerModelConfig,
                                         jax_index_select, is_power_of_two)


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: Any = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


@dataclass
class InferenceFuncConfig:
    """Implements a minimal config class for using huggingface's generator.

    Note: these parameters might be overwritten by model.generate(**kwargs).
    """
    bos_token_id: int = 0
    num_beams: int = 1
    num_beam_groups: int = 1
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    early_stopping: bool = False
    num_return_sequences: int = 1
    pad_token_id: int = 1
    eos_token_id: int = 2
    output_scores: bool = False
    output_attentions: bool = False
    output_hidden_states: bool = False
    return_dict_in_generate: bool = False
    is_encoder_decoder: bool = False
    min_length: bool = 0
    no_repeat_ngram_size: int = 0
    encoder_no_repeat_ngram_size: int = 0
    bad_words_ids: Sequence = None
    diversity_penalty: float = 0.0
    forced_bos_token_id: int = None
    forced_eos_token_id: int = None
    remove_invalid_values: bool = False
    exponential_decay_length_penalty: float = None
    do_sample: bool = False
    top_k: int = 50
    top_p: int = 1.0
    typical_p: int = 1.0
    temperature: float = 1.0


class WrappedInferenceFunc(GenerationMixin):
    """
    Wrap an inference func as a GenerationMixin.
    This class implements the minimal interface for using huggingface's generator.
    """

    def __init__(self, inference_func, config, executable, transformer_config):
        self.inference_func = inference_func
        self.config = config
        self.main_input_name = "input_ids"
        self.executable = executable  # An alpa executable
        self.transformer_config = transformer_config
        self.index_select_executables = {}
        self.cache_location = None

    def forward(self, attention_mask):
        # This function is never used
        raise NotImplementedError()

    def prepare_inputs_for_generation(self, input_ids, attention_mask,
                                      past=None, **kwargs):
        # If past is defined, it means we are in the decoding stage,
        # so we only process the last token
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        ret = {"input_ids": input_ids, "past_key_values": past,
               "attention_mask": attention_mask}
        return ret

    def __call__(self,
                 input_ids,
                 past_key_values=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 attention_mask=None,
                 return_dict=None):
        ret = self.inference_func(input_ids,
                                  past_key_values,
                                  attention_mask=attention_mask,
                                  output_hidden_states=output_hidden_states,
                                  output_attentions=output_attentions)
        return ret

    def _reorder_cache(self, past, beam_idx):
        # Reorder cache for beam search

        # PyTorch
        if hasattr(past[0][0], "index_select"):
            return tuple(
                tuple(
                    past_state.index_select(0, beam_idx)
                    for past_state in layer_past)
                for layer_past in past)

        # Jax (single-device)
        if not isinstance(past[0][0], DistributedArray):
            beam_idx = jnp.array(beam_idx.to("cpu").numpy())
            return tuple(
                tuple(
                    jax_index_select(past_state, beam_idx, 0)
                    for past_state in layer_past)
                for layer_past in past)

        # Alpa
        mesh_groups = defaultdict(list)
        if self.cache_location is None:
            self.cache_location = []
            for layer_past in past:
                tmp_loc = []
                for past_state in layer_past:
                    assert isinstance(past_state, DistributedArray)
                    mesh = past_state.device_mesh
                    mesh_groups[mesh].append(past_state)
                    tmp_loc.append((mesh, len(mesh_groups[mesh]) - 1))
                self.cache_location.append(tmp_loc)
        else:
            for layer_past in past:
                for past_state in layer_past:
                    assert isinstance(past_state, DistributedArray)
                    mesh = past_state.device_mesh
                    mesh_groups[mesh].append(past_state)

        beam_idx = beam_idx.to("cpu").numpy()

        def grouped_reorder_cache(arys, device_mesh):
            if len(arys) == 0:
                return []
            if device_mesh in self.index_select_executables:
                executable = self.index_select_executables[device_mesh]
            else:
                dim = 0
                avals = [ary.aval for ary in arys]
                specs = [ary.sharding_spec for ary in arys]
                executable = get_index_select_mesh_executable(
                    avals, specs, beam_idx, dim, device_mesh,
                    [False] * len(avals))
                self.index_select_executables[device_mesh] = executable
            ret = executable(*arys, beam_idx)
            for v in ret:
                v.skip_shard_args_check = True
            return ret

        results = {
            mesh: grouped_reorder_cache(mesh_groups[mesh], mesh)
            for mesh in mesh_groups
        }

        return tuple(
            tuple(results[mesh][loc]
                  for mesh, loc in layer_loc)
            for layer_loc in self.cache_location)


def get_hf_opt_model(model_name, device, num_beams):
    raw_model = OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    raw_model = raw_model.to(device)

    def inference_func(input_ids,
                       past_key_values,
                       attention_mask,
                       output_attentions,
                       output_hidden_states):
        out = raw_model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states)
        return InferenceFuncOutput(out.logits, out.past_key_values)

    inference_func_config = InferenceFuncConfig(num_beams=num_beams)
    for key in inference_func_config.__dataclass_fields__.keys():
        setattr(inference_func_config, key, getattr(raw_model.config, key))
    transformer_config = TransformerModelConfig(
        H=raw_model.config.hidden_size,
        L=raw_model.config.num_hidden_layers,
        n_head=raw_model.config.num_attention_heads,
        seq_len=raw_model.config.max_position_embeddings,
        vocab_size=raw_model.config.vocab_size)
    executable = None
    return WrappedInferenceFunc(inference_func, inference_func_config,
                                executable, transformer_config)


def get_model(model_name: str,
              device: str,
              # Weights
              path: str,
              dummy: bool = False,
              # Model parameters
              autoregressive: bool = True,
              dtype=jnp.float16,
              # Batch size and seq length
              batch_size: int = 1,
              num_micro_batches: int = 1,
              max_target_positions: int = 2048,
              encoder_seq_lengths: Sequence[int] = [1],
              # Shared arguments with model.generate
              do_sample: bool = False,
              num_beams: int = 1,
              num_return_sequences: int = 1,
              return_dict_in_generate: bool = True,
              output_attentions: bool = False,
              output_hidden_states: bool = False):
    """Get a model that is compatible with HuggingFace's generation API.

    Args:
        model_name: "facebook/opt-", or "alpa/opt-".
        device: "cpu" or "gpu". This only controls the device used
          by pytorch. Alpa always runs on GPU.
        path: The path to opt weights.
        dummy: Use dummy weights for faster debugging.
        encoder_seq_lengths: compile mutliple executables for multiple
          encoder sequence lengths.
    """
    if not model_name.startswith("alpa") and not autoregressive:
        raise NotImplementedError(
            f"Cannot support {model_name} in forward-only mode.")
    if autoregressive and num_micro_batches > 1:
        raise NotImplementedError(
            f"Cannot support num_micro_batches > 1 in autoregressive mode.")

    if "facebook/opt" in model_name:
        return get_hf_opt_model(model_name, device, num_beams)

    assert ("jax/opt" in model_name or "alpa/opt" in model_name)
    assert return_dict_in_generate

    if autoregressive and 1 not in encoder_seq_lengths:
        encoder_seq_lengths += [1]
    encoder_seq_lengths.sort()

    # weight path
    name = model_name.split("-")[1].upper()
    path = os.path.join(path, f"{name}_np")
    if not dummy:
        assert os.path.exists(path), f"No such file or directory: '{path}'"

    # figure out the actual input size
    if do_sample:
        expand_size = batch_size * num_beams * num_return_sequences
    else:
        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`."
            )
        expand_size = batch_size * num_beams

    if "jax/opt" in model_name:
        config = get_opt_config(name,
                                num_pp_stages=None,
                                mark_boundary=False,
                                dtype=dtype,
                                max_target_positions=max_target_positions)
        transformer_config = TransformerModelConfig(
            H=config.decoder_embed_dim,
            L=config.decoder_layers,
            n_head=config.decoder_attention_heads,
            seq_len=config.max_target_positions,
            vocab_size=config.vocab_size)

        executables, params_aval = get_jax_executable(
            config, encoder_seq_lengths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        # load params
        params = load_params_np(params_aval, path, config, dummy)
        init_cache = init_cache_np(config, batch_size=expand_size)
        params, init_cache = jax.tree_map(jnp.array, (params, init_cache))
    else:
        assert "alpa/opt" in model_name
        assert is_power_of_two(num_beams), "num_beams must be a power of two"
        alpa.init()
        alpa.global_config.xla_client_mem_fraction = 0.88

        print(
            f"Load model {model_name} ... (This can take several minutes for very large models)"
        )

        num_pp_stages = max(2, alpa.get_global_cluster().num_hosts)
        num_pp_stages = min(num_pp_stages,
                            alpa.get_global_cluster().num_devices)
        config = get_opt_config(name,
                                num_pp_stages=num_pp_stages,
                                dtype=dtype,
                                max_target_positions=max_target_positions)
        transformer_config = TransformerModelConfig(
            H=config.decoder_embed_dim,
            L=config.decoder_layers,
            n_head=config.decoder_attention_heads,
            seq_len=config.max_target_positions,
            vocab_size=config.vocab_size)

        executables, params_aval = get_pipeshard_executable(
            config,
            batch_size=expand_size,
            num_micro_batches=num_micro_batches,
            encoder_seq_lengths=encoder_seq_lengths,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            autoregressive=autoregressive)

        # Load params
        print(" - Load parameters""")
        params = load_multi_executable_params_dis_array(
            path, executables, params_aval, config, dummy)

        if autoregressive:
            init_cache = init_multi_executable_cache_dis_array(executables,
                                                               config,
                                                               expand_size,
                                                               dummy=dummy)
            set_skip_shard_args_check(init_cache)

        for executable in executables.values():
            executable.sync()

        # return executable directly if not autoregressive
        if not autoregressive:
            assert len(executables) == 1
            return list(
                executables.values())[0], params, transformer_config

    num_valid_tokens = None
    last_token = None
    step_ct = 0

    def inference_func(input_ids,
                       past_key_values,
                       attention_mask,
                       output_attentions,
                       output_hidden_states):
        input_ids = input_ids.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()

        def run_one(_executable, _input_ids, _past_key_values, _attention_mask, num_internal_pad):
            nonlocal num_valid_tokens
            nonlocal last_token
            nonlocal step_ct

            if _past_key_values is None:
                # Init all states
                _past_key_values = init_cache
                num_valid_tokens = np.zeros((expand_size, 1), dtype=np.int32)
                last_token = np.zeros((expand_size, 1), dtype=np.int32)
                step_ct = 0

            if _input_ids.shape[1] == 1:
                # A fast path for step_len = 1
                cum_sum = _attention_mask[:, -1:]
                num_valid_tokens = num_valid_tokens + cum_sum
                position_ids_step = num_valid_tokens + config.pad
                last_token = np.where(cum_sum, _input_ids, last_token)
                _input_ids = last_token
            else:
                # A general path that works for any step_len
                cumsum = np.cumsum(_attention_mask[:,step_ct:], axis=1, dtype=np.int32)
                position_ids_step = num_valid_tokens + cumsum + config.pad
                num_valid_tokens_step = cumsum[:,-1:]
                num_valid_tokens = num_valid_tokens + num_valid_tokens_step

                last_token = np.where(num_valid_tokens_step > 0,
                     np.take_along_axis(_input_ids, num_valid_tokens_step - 1, axis=1),
                     last_token)
                _input_ids = np.where(_attention_mask[:, step_ct:], _input_ids, last_token)

            # Use value "2" as a special mask to represent internal padding
            if num_internal_pad:
                _attention_mask[:,-num_internal_pad:] = 2
            _attention_mask = pad_attention_mask(_attention_mask, max_target_positions)

            output = _executable(
                params, {
                    "input_ids": _input_ids,
                    "position_ids": position_ids_step,
                    "cache": _past_key_values,
                    "mask": _attention_mask,
                })

            step_ct += _input_ids.shape[1] - num_internal_pad
            set_skip_shard_args_check(output.attention_cache)

            return output

        seq_len = input_ids.shape[1]
        if seq_len == 1:
            # A fast path for seq_len = 1
            output = run_one(executables[1], input_ids, past_key_values, attention_mask, 0)
        else:
            # A general path that works for all seq_len
            i = 0
            while i < seq_len:
                remaining = seq_len - i
                step_len = get_padded_step_len(remaining, encoder_seq_lengths)

                step_input_ids = input_ids[:, i:i + step_len]
                step_attention_mask = (
                    attention_mask[:, :attention_mask.shape[1] - remaining + step_len])

                if step_input_ids.shape[1] != step_len:
                    # Pad the inputs and masks to step_len
                    # Note that this kind of internal padding is different from
                    # the padding added by the tokenizer. This internal padding
                    # should not update cache and step_ct
                    num_internal_pad = step_len - step_input_ids.shape[1]
                    pad_shape = (expand_size, num_internal_pad)
                    step_input_ids = np.concatenate(
                        (step_input_ids, np.zeros(pad_shape, dtype=np.int32)), axis=1)
                    step_attention_mask = np.concatenate(
                        (step_attention_mask, np.zeros(pad_shape, dtype=np.int8)), axis=1)
                else:
                    num_internal_pad = 0

                output = run_one(executables[step_len], step_input_ids,
                                 past_key_values, step_attention_mask,
                                 num_internal_pad)
                past_key_values = output.attention_cache
                i += step_input_ids.shape[1]

        logits_step = torch.from_numpy(np.array(output.logits)).to(device)
        return InferenceFuncOutput(logits_step, output.attention_cache,
                                   output.hidden_states, output.attentions)

    inference_func_config = InferenceFuncConfig(num_beams=num_beams)
    return WrappedInferenceFunc(inference_func,
                                inference_func_config,
                                executables[1],
                                transformer_config)


def get_padded_step_len(length, encoder_seq_lengths):
    """For a given length, find the smallest value in encoder_seq_lengths that
    is greater than the given length."""
    for i in range(len(encoder_seq_lengths)):
        if encoder_seq_lengths[i] >= length:
            break
    return encoder_seq_lengths[i]


def set_skip_shard_args_check(attention_cache):
    """
    Skip the check in DistributedPhysicalDeviceMesh::shard_args for
    attention cache. We need this hack because attention_cache is
    a batch var but alpa doesn't implement a fast path for batch vars.
    """
    if isinstance(attention_cache[0], alpa.device_mesh.DistributedArray):
        for x in attention_cache:
            x.skip_shard_args_check = True
    else:
        for y in attention_cache:
            for x in y:
                if isinstance(x, alpa.device_mesh.DistributedArray):
                    x.skip_shard_args_check = True


def pad_attention_mask(mask, max_target_positions):
    """Pad attention mask to the shape [B, 1, 1, max_target_positions]. """
    batch_size = mask.shape[0]
    ret_mask = np.zeros((batch_size, max_target_positions), dtype=np.int8)
    ret_mask[:, :mask.shape[-1]] = mask
    ret_mask = ret_mask[:, np.newaxis, np.newaxis, :]
    return ret_mask
