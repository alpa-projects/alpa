"""Wrap models to make them compatible with huggingface's generator API."""
from collections import defaultdict
import os
import time
from typing import Sequence, Any, Optional
import warnings

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
from transformers import OPTForCausalLM, BloomForCausalLM, GPT2LMHeadModel
from tqdm import tqdm

from llm_serving.model import opt_model, bloom_model
from llm_serving.model.opt_utils import (TransformerModelConfig,
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


def get_hf_model(model_name, device):
    """Get a huggingface model."""
    disable_torch_init()
    if "opt" in model_name:
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        model_class = BloomForCausalLM
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    model = model_class.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if "cuda" in device else torch.float32)
    model = model.to(device)
    restore_torch_init()

    def inference_func(input_ids,
                       past_key_values,
                       attention_mask,
                       output_attentions,
                       output_hidden_states):
        out = model(input_ids=input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states)
        return InferenceFuncOutput(out.logits, out.past_key_values)

    inference_func_config = InferenceFuncConfig()
    for key in inference_func_config.__dataclass_fields__.keys():
        setattr(inference_func_config, key, getattr(model.config, key))
    if hasattr(model.config, "seq_length"):
        seq_len = model.config.seq_length
    else:
        seq_len = model.config.max_position_embeddings

    transformer_config = TransformerModelConfig(
        H=model.config.hidden_size,
        L=model.config.num_hidden_layers,
        n_head=model.config.num_attention_heads,
        seq_len=seq_len,
        vocab_size=model.config.vocab_size)
    executable = None
    return WrappedInferenceFunc(inference_func, inference_func_config,
                                executable, transformer_config)


def get_alpa_model(model_name: str,
                   # Weights
                   path: str,
                   dummy: bool = False,
                   # Batch size and seq length
                   batch_size: int = 1,
                   num_micro_batches: int = 1,
                   max_seq_len: int = 2048,
                   encoder_chunk_sizes: Sequence[int] = (1, 64),
                   num_pp_stages: Optional[int] = None,
                   # Model parameters
                   dtype=jnp.float16,
                   torch_device: str = "cpu",
                   # Shared arguments with model.generate
                   do_sample: bool = False,
                   num_beams: int = 1,
                   num_return_sequences: int = 1,
                   return_dict_in_generate: bool = True,
                   output_attentions: bool = False,
                   output_hidden_states: bool = False):
    """Get a alpa-based model that is compatible with HuggingFace's generation API."""
    if num_micro_batches > 1:
        raise NotImplementedError()
    assert return_dict_in_generate

    if 1 not in encoder_chunk_sizes:
        encoder_chunk_sizes += [1]
    encoder_chunk_sizes = list(set(encoder_chunk_sizes))
    encoder_chunk_sizes.sort()

    # weight path
    name = model_name.split("/")[1].lower()
    path = os.path.abspath(os.path.expanduser(os.path.join(path, f"{name}-np")))
    if not dummy:
        # Download weights if there is no cached weights.
        if not os.path.exists(path):
            if name in ["opt-175b"]:
                raise ValueError(f"Cannot find cached weights under '{path}'. "
                                  "Please follow the instructions to download "
                                  "and convert weights manually. ")
            print(f"Cannot find cached weights under '{path}'.")
            download_weights(model_name.split("/")[1], path)

        # Do some sanity check
        assert os.path.exists(path), f"No such file or directory: '{path}'"
        if "opt" in name:
            embed_weight = os.path.join(path, "decoder.embed_tokens.weight")
        elif "bloom" in name:
            embed_weight = os.path.join(path, "word_embeddings.weight")
        assert os.path.exists(embed_weight), f"No such file or directory: '{embed_weight}'"

    # Figure out the actual input size
    if do_sample:
        batch_size = batch_size * num_beams * num_return_sequences
    else:
        if num_return_sequences > num_beams:
            raise ValueError(
                "`num_return_sequences` has to be smaller or equal to `num_beams`."
            )
        batch_size = batch_size * num_beams

    if "jax" in model_name:
        if "opt" in model_name:
            m = opt_model
        elif "bloom" in model_name:
            m = bloom_model
            if any(x > 1 for x in encoder_chunk_sizes):
                # TODO: support chunk size > 1
                warnings.warn("encoder_chunk_size > 1 is not supported. Ignored.")
                encoder_chunk_sizes = [1]
        config = m.get_config(name,
                              num_pp_stages=None,
                              mark_boundary=False,
                              dtype=dtype,
                              max_seq_len=max_seq_len)
        transformer_config = TransformerModelConfig(
            H=config.hidden_size,
            L=config.num_hidden_layers,
            n_head=config.n_head,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size)

        executables, params_aval = m.get_jax_executable(
            config, encoder_chunk_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)

        # load params
        params = m.load_params_np(params_aval, path, config, dummy)
        init_cache = m.init_cache_np(config, batch_size=batch_size)
        params, init_cache = jax.tree_map(jnp.array, (params, init_cache))
    elif "alpa" in model_name:
        if "opt" in model_name:
            m = opt_model
        elif "bloom" in model_name:
            m = bloom_model
            if any(x > 1 for x in encoder_chunk_sizes):
                # TODO: support chunk size > 1
                warnings.warn("encoder_chunk_size > 1 is not supported. Ignored.")
                encoder_chunk_sizes = [1]

        alpa.init()

        print(
            f"Load model {model_name} ... "
            f"(This can take several minutes for very large models)"
        )

        if num_pp_stages is None:
            num_pp_stages = max(2, alpa.get_global_cluster().num_hosts)
            num_pp_stages = min(num_pp_stages,
                                alpa.get_global_cluster().num_devices)
        config = m.get_config(name,
                              num_pp_stages=num_pp_stages,
                              dtype=dtype,
                              max_seq_len=max_seq_len)
        transformer_config = TransformerModelConfig(
            H=config.hidden_size,
            L=config.num_hidden_layers,
            n_head=config.n_head,
            seq_len=config.max_seq_len,
            vocab_size=config.vocab_size)

        print(f" - Compile executables for encoder_chunk_sizes={encoder_chunk_sizes}. ",
              end="", flush=True)
        tic = time.time()
        executables, params_aval = m.get_pipeshard_executable(
            config,
            batch_size=batch_size,
            num_micro_batches=num_micro_batches,
            encoder_chunk_sizes=encoder_chunk_sizes,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        print(f"elapsed: {time.time() - tic:.2f} second.")

        # Load params
        print(" - Load parameters. ", end="", flush=True)
        tic = time.time()
        params = m.load_multi_executable_params_dis_array(
            path, executables, params_aval, config, dummy)

        init_cache = m.init_multi_executable_cache_dis_array(
            executables, config, batch_size, dummy=dummy)
        set_skip_shard_args_check(init_cache)

        for executable in executables.values():
            executable.sync()
        print(f"elapsed: {time.time() - tic:.2f} second.")
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    num_valid_tokens = None
    last_token = None
    step_ct = 0

    def inference_func(input_ids,
                       past_key_values,
                       attention_mask,
                       output_attentions,
                       output_hidden_states):
        assert input_ids.shape[0] == batch_size, (
            f"Expect batch size = {batch_size}, but got {input_ids.shape[0]}")
        input_ids = input_ids.cpu().numpy()
        attention_mask = attention_mask.cpu().numpy()

        def run_one(_executable, _input_ids, _past_key_values, _attention_mask, num_internal_pad):
            nonlocal num_valid_tokens
            nonlocal last_token
            nonlocal step_ct

            if _past_key_values is None:
                # Init all states
                _past_key_values = init_cache
                num_valid_tokens = np.zeros((batch_size, 1), dtype=np.int32)
                last_token = np.zeros((batch_size, 1), dtype=np.int32)
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

            if num_internal_pad:
                # Use value "2" as a special mask to represent internal padding
                _attention_mask[:,-num_internal_pad:] = 2
            _attention_mask = pad_attention_mask(_attention_mask, max_seq_len)

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
        if seq_len == 1:  # A fast path for seq_len = 1
            output = run_one(executables[1], input_ids, past_key_values, attention_mask, 0)
        else:  # A general path that works for all seq_len
            i = 0
            while i < seq_len:
                remaining = seq_len - i
                step_len = get_padded_step_len(remaining, encoder_chunk_sizes)

                step_input_ids = input_ids[:, i:i + step_len]
                step_attention_mask = (
                    attention_mask[:, :attention_mask.shape[1] - remaining + step_len])

                if step_input_ids.shape[1] != step_len:
                    # Pad the inputs and masks to step_len
                    # Note that this kind of internal padding is different from
                    # the padding added by the tokenizer. This internal padding
                    # should not update cache and step_ct
                    num_internal_pad = step_len - step_input_ids.shape[1]
                    pad_shape = (batch_size, num_internal_pad)
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

        logits_step = torch.from_numpy(np.array(output.logits)).to(torch_device).float()
        return InferenceFuncOutput(logits_step, output.attention_cache,
                                   output.hidden_states, output.attentions)

    inference_func_config = InferenceFuncConfig()
    return WrappedInferenceFunc(inference_func,
                                inference_func_config,
                                executables[1],
                                transformer_config)


def get_model(model_name: str,
              # Weights
              path: str,
              dummy: bool = False,
              # Batch size and seq length
              batch_size: int = 1,
              num_micro_batches: int = 1,
              max_seq_len: int = 2048,
              encoder_chunk_sizes: Sequence[int] = (1, 64),
              num_pp_stages: Optional[int] = None,
              # Model parameters
              dtype=jnp.float16,
              torch_device: str = "cpu",
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
        path: The path to opt weights.
        dummy: Use dummy weights for faster debugging.
        batch_size: The batch size.
        num_micro_batches: The number of micro batch sizs in pipeline
          parallelism.
        max_seq_len: The max sequence length.
        encoder_chunk_sizes: Compile mutliple executables with different
          chunk sizes. These executables are used to encoding prompts
          chunk by chunk.
        num_pp_stages: The number of pipeline parallelism stages.
        dtype: The type of parameters.
        torch_device: "cpu" or "gpu". This only controls the device used
          by pytorch. Alpa always runs on GPU.
        other parameters: shared with huggingface's model.generate API.
    """
    if "facebook/opt" in model_name or "bigscience/bloom" in model_name:
        return get_hf_model(model_name, torch_device)
    elif ("jax/opt" in model_name or "alpa/opt" in model_name or
          "jax/bloom" in model_name or "alpa/bloom" in model_name):
        return get_alpa_model(
              model_name,
              path,
              dummy,
              batch_size,
              num_micro_batches,
              max_seq_len,
              encoder_chunk_sizes,
              num_pp_stages,
              dtype,
              torch_device,
              do_sample,
              num_beams,
              num_return_sequences,
              return_dict_in_generate,
              output_attentions,
              output_hidden_states)
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def get_padded_step_len(length, encoder_chunk_sizes):
    """For a given length, find the smallest value in encoder_chunk_sizes that
    is greater than the given length."""
    for i in range(len(encoder_chunk_sizes)):
        if encoder_chunk_sizes[i] >= length:
            break
    return encoder_chunk_sizes[i]


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


def pad_attention_mask(mask, max_seq_len):
    """Pad attention mask to the shape [B, 1, 1, max_seq_len]. """
    batch_size = mask.shape[0]
    ret_mask = np.zeros((batch_size, max_seq_len), dtype=np.int8)
    ret_mask[:, :mask.shape[-1]] = mask
    ret_mask = ret_mask[:, np.newaxis, np.newaxis, :]
    return ret_mask


def download_weights(model_name, path):
    """Download weights from huggingface."""
    if "opt" in model_name:
        hf_model_name = "facebook/" + model_name
        model_class = OPTForCausalLM
    elif "bloom" in model_name:
        hf_model_name = "bigscience/" + model_name
        model_class = BloomForCausalLM

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to alpa format under {path} ...")
    if "opt" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            name = name.replace("decoder.final_layer_norm", "decoder.layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    elif "bloom" in model_name:
        for name, param in tqdm(list(model.transformer.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)
