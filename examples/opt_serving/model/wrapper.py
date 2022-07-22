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

from opt_serving.model.opt_model import (
    get_opt_config, get_pipeshard_executable, load_params_dis_array,
    init_cache_dis_array, load_params_np, init_cache_np, get_jax_executable)
from opt_serving.model.opt_utils import (TransformerModelConfig,
                                                  jax_index_select,
                                                  is_power_of_two)


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: Any = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


@dataclass
class InferenceFuncConfig:
    """Implements a minimal config class for using huggingface's generator.

    Note: these paramerers might be overwritten by model.generate(**kwargs).
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
    top_k: int = 50
    top_p: int = 1.0
    typical_p: int = 1.0
    temperature: float = 1.0


class WrappedInferenceFunc(GenerationMixin):
    """
    Wrap an inference func as a GenerationMixin.
    This class implements the minimal interface for using huggingface's generator.

    This class also decomposes the first call of prompt during generation to one token by one token.
    """

    def __init__(self, inference_func, config, executable, transformer_config):
        self.inference_func = inference_func
        self.config = config
        self.main_input_name = "input_ids"
        self.executable = executable
        self.transformer_config = transformer_config
        self.index_select_executables = {}
        self.cache_location = None

    def forward(self, attention_mask):
        # This function is never used
        raise NotImplementedError()

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        # only last token for input_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {
            "input_ids": input_ids,
            "past_key_values": past,
        }

    def __call__(self,
                 input_ids,
                 past_key_values=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 return_dict=None):
        # Decompose the call to token by token
        for i in range(input_ids.shape[1]):
            ret = self.inference_func(input_ids[:, i:i + 1],
                                      past_key_values,
                                      output_hidden_states=output_hidden_states,
                                      output_attentions=output_attentions)
            past_key_values = ret.past_key_values
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


def get_hf_gpt_model(model_name, device, num_beams):
    raw_model = GPT2LMHeadModel.from_pretrained(model_name)
    raw_model = raw_model.to(device)

    def inference_func(input_ids,
                       past_key_values,
                       output_attentions=False,
                       output_hidden_states=False):
        out = raw_model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states)
        return InferenceFuncOutput(out.logits, out.past_key_values)

    inference_func_config = raw_model.config
    inference_func_config.num_beams = num_beams
    transformer_config = TransformerModelConfig(
        H=raw_model.config.n_embd,
        L=raw_model.config.n_layer,
        n_head=raw_model.config.n_head,
        seq_len=raw_model.config.n_positions,
        vocab_size=raw_model.config.vocab_size)
    executable = None
    return WrappedInferenceFunc(inference_func, inference_func_config,
                                executable, transformer_config)


def get_hf_opt_model(model_name, device, num_beams):
    raw_model = OPTForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    raw_model = raw_model.to(device)

    def inference_func(input_ids,
                       past_key_values,
                       output_attentions=False,
                       output_hidden_states=False):
        if past_key_values is None:
            attention_mask = None
        else:
            past_length = past_key_values[0][0].shape[2]
            attention_mask = torch.ones(
                (input_ids.shape[0], past_length + 1)).to(device)
        out = raw_model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
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
              path: str,
              autoregressive=True,
              dtype=jnp.float16,
              dummy=False,
              batch_size=1,
              num_beams=1,
              decoding_length_per_step=1,
              num_micro_batches=1,
              support_output_attentions=False,
              support_output_hidden_states=False):
    """Get and load model and return a WrappedInferenceFunc compatible with HuggingFace.

    Args:
        model_name: "gpt", "facebook/opt-", or "alpa/opt-".
    """
    if not model_name.startswith("alpa") and not autoregressive:
        raise NotImplementedError(
            f"Cannot support {model_name} in forward-only mode.")
    if autoregressive and decoding_length_per_step > 1:
        raise RuntimeError(
            f"Autoregressive requires decoder_length_per_step == 1")
    if autoregressive and num_micro_batches > 1:
        raise NotImplementedError(
            f"Cannot support num_micro_batches > 1 in autoregressive mode.")

    if "gpt" in model_name:
        return get_hf_gpt_model(model_name, device, num_beams)
    if "facebook/opt" in model_name:
        return get_hf_opt_model(model_name, device, num_beams)

    assert ("jax/opt" in model_name or "alpa/opt" in model_name)
    name = model_name.split("-")[1].upper()

    # weight path
    path = os.path.join(path, f"{name}_np")
    if not dummy:
        assert os.path.exists(path), f"No such file or directory: '{path}'"

    if "jax/opt" in model_name:
        config = get_opt_config(name,
                                num_pp_stages=None,
                                mark_boundary=False,
                                dtype=dtype)
        transformer_config = TransformerModelConfig(
            H=config.decoder_embed_dim,
            L=config.decoder_layers,
            n_head=config.decoder_attention_heads,
            seq_len=config.max_target_positions,
            vocab_size=config.vocab_size)

        executable, params_aval = get_jax_executable(
            config,
            support_output_attentions=support_output_attentions,
            support_output_hidden_states=support_output_hidden_states)

        # load params
        params = load_params_np(params_aval, path, config, dummy)
        init_cache = init_cache_np(config, batch_size=batch_size * num_beams)
        params, init_cache = jax.tree_map(jnp.array, (params, init_cache))
    else:
        assert "alpa/opt" in model_name
        assert is_power_of_two(num_beams), "num_beams must be a power of two"
        alpa.init()

        print(
            f"Load model {model_name} ... (This can take several minutes for very large models)"
        )

        num_pp_stages = max(2, alpa.get_global_cluster().num_hosts)
        num_pp_stages = min(num_pp_stages,
                            alpa.get_global_cluster().num_devices)
        config = get_opt_config(name, num_pp_stages=num_pp_stages, dtype=dtype)
        transformer_config = TransformerModelConfig(
            H=config.decoder_embed_dim,
            L=config.decoder_layers,
            n_head=config.decoder_attention_heads,
            seq_len=config.max_target_positions,
            vocab_size=config.vocab_size)

        if autoregressive:
            assert batch_size == 1, "we only support batch_sie = 1 for autoregressive!"
        executable, params_aval = get_pipeshard_executable(
            config,
            batch_size=batch_size * num_beams,
            num_micro_batches=num_micro_batches,
            decoding_length_per_step=decoding_length_per_step,
            support_output_attentions=support_output_attentions,
            support_output_hidden_states=support_output_hidden_states,
            autoregressive=autoregressive)

        # load params
        params = load_params_dis_array(path, executable, params_aval, config,
                                       dummy)
        if autoregressive:
            init_cache = init_cache_dis_array(executable,
                                              config,
                                              batch_size * num_beams,
                                              dummy=dummy)
            set_skip_shard_args_check(init_cache)
        executable.sync()

        # return executable directly if not autoregressive
        if not autoregressive:
            return executable, params, transformer_config

    step_ct = 0

    def inference_func(input_ids,
                       past_key_values,
                       output_attentions=False,
                       output_hidden_states=False):
        nonlocal step_ct

        if past_key_values is None:
            past_key_values = init_cache
            step_ct = 0

        input_ids_step = input_ids.cpu().numpy()
        position_ids_step = np.full_like(input_ids_step,
                                         step_ct + config.pad + 1)

        output = executable(
            params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": past_key_values,
            })
        set_skip_shard_args_check(output.attention_cache)

        logits_step = torch.from_numpy(np.array(output.logits)).to(device)

        step_ct += 1
        return InferenceFuncOutput(logits_step, output.attention_cache,
                                   output.hidden_states, output.attentions)

    inference_func_config = InferenceFuncConfig(num_beams=num_beams)
    return WrappedInferenceFunc(inference_func, inference_func_config,
                                executable, transformer_config)


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
