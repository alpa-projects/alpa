from typing import Union, List

import numpy as np
import os

import jax
import jax.numpy as jnp
import torch
import tqdm
from llm_serving.model import opt_model_1d

from alpa.timer import timers
from transformers import OPTForCausalLM, BloomForCausalLM
from transformers.generation_utils import dataclass
from examples.llm_serving.model.opt_utils import TransformerModelConfig
from examples.llm_serving.model.wrapper import disable_torch_init, restore_torch_init
from examples.llm_serving.model import opt_model
from examples.llm_serving.model.opt_model_1d import IterationLevelInputPool, BatchLevelInputPool, unpad


@dataclass
class InputPoolConfig:
    """The config for iterative-level input pool."""
    batch_size: int = 512
    cache_size: int = 4096
    max_cache_per_seq: int = 128


class SequenceGenerator:
    def __init__(self, executable, params, input_pool_config, inference_func_config):
        self.executable = executable
        self.params = params
        self.input_pool_config = input_pool_config
        self.inference_func_config = inference_func_config

        # some other attributes:
        self.pad = self.inference_func_config.pad_token_id

    def generate(self,
                 input: Union[IterationLevelInputPool, List[List[int]], np.ndarray],
                 max_length=64,
                 do_sample=False,
                 **kwargs):
        generate_args = {
            "max_length": max_length,
            "do_sample": do_sample
        }
        for key, val in kwargs.items():
            generate_args[key] = val
        if isinstance(input, IterationLevelInputPool):
            raise NotImplementedError()
            # return self.generate_by_stream(input, **generate_args)
        elif isinstance(input, (List, np.ndarray, torch.Tensor)):
            unpadded_input = unpad(input)
            return self.generate_by_batch(unpadded_input, **generate_args)
        else:
            raise RuntimeError()

    def generate_by_batch(self, input_ids: List[List[int]]):
        input_pool = IterationLevelInputPool(self.model_config,
                                             self.input_pool_config.batch_size,
                                             self.input_pool_config.cache_size,
                                             self.input_pool_config.max_cache_per_seq)
        input_pool.enter_prompts(input_ids)

        while not input_pool.is_finished():
            input, input_index, position_ids = input_pool.get_next_iter_input()

            input_pool.set_environments()
            batch = {
                "input_ids": input,
                "position_ids": position_ids,
                "cache": input_pool.kv_caches
            }

            # compute
            logits, kv = self.executable(self.params, batch)

            generated_ids = self._generate_greedy(logits, [len(seq) for seq in input_ids])
            # update cache
            input_pool.update_cache(kv, generated_ids)

        ret = input_pool.get_results()
        # post-process generation results
        return ret

    @staticmethod
    def _generate_greedy(logits, seq_lens):
        outputs = []
        next_token = np.array(jnp.argmax(logits, axis=-1))
        output_idx = -1
        for seq_len in seq_lens:
            output_idx += seq_len
            outputs.append([int(next_token[output_idx])])
        return outputs


def get_model_1d(model_name: str,
                 path: str,
                 dummy: bool = False,
                 # batch size, this batch is #tokens
                 batch_size: int = 256,
                 max_seq_len: int = 2048,
                 cache_size: int = 4096,
                 max_cache_per_seq: int = 128,
                 # model parameters
                 dtype=jnp.float16,
                 torch_device: str = "cpu",
                 # Shared arguments with model.generate
                 do_sample: bool = False,
                 num_beams: int = 1,
                 num_return_sequences: int = 1,
                 return_dict_in_generate: bool = True,
                 output_attentions: bool = False,
                 output_hidden_states: bool = False):
    """Experimental 1D transformer implementation."""
    assert "opt-1d" in model_name, "are you sure you want to use the experimental 1D version?"
    name = model_name.split("/")[1].lower()
    name = name.replace("-1d", "")
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
    # TODO(Hao): figure out the actual input size
    config = opt_model.get_config(name, dtype=dtype, max_seq_len=max_seq_len)
    transformer_config = TransformerModelConfig(
        H=config.hidden_size,
        L=config.num_hidden_layers,
        n_head=config.n_head,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size)
    executable, params_aval = opt_model_1d.get_jax_executable(
        config,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states)

    # load params
    # TODO(Hao): use the same func with 2D
    params = opt_model_1d.load_params_np(params_aval, path, config, dummy)
    params = jax.tree_map(jnp.array, params)
    input_pool = BatchLevelInputPool(config, batch_size=batch_size, cache_size=cache_size,
                                     max_cache_per_seq=max_cache_per_seq)

    def sync(device_id=0):
        jax.devices()[device_id].synchronize_all_activity()
        return

    def inference_func(input_ids,
                       past_key_values,
                       attention_mask,
                       output_attentions,
                       output_hidden_states):
        timers("enter").start(sync)
        if input_ids.shape[1] > 1:
            unpadded_input = input_pool.unpad(input_ids)
            input, input_index, position_ids = input_pool.enter_prompts(unpadded_input)
        else:
            unpadded_input = input_ids.tolist()
            input, input_index, position_ids = input_pool.enter_decoding(unpadded_input)
        timers("enter").suspend(sync)

        # set envvar
        os.environ["FT_INPUT_INDEX_ADDR"] = str(input_index.ctypes.data)
        os.environ["FT_CACHE_INDEX_ADDR"] = str(input_pool.kv_cache_ids.ctypes.data)
        os.environ["FT_MAX_CACHE_LEN_PER_SEQ"] = str(input_pool.max_cache_per_seq)

        batch = {
            "input_ids": input,
            "position_ids": position_ids,
            "cache": input_pool.kv_caches
        }
        timers("compute").start(sync)
        logits, kv = executable(params, batch)
        timers("compute").suspend(sync)

        timers("update").start(sync)
        input_pool.update_cache(unpadded_input, kv)
        input_pool.update_num_prev_tokens(unpadded_input)
        timers("update").suspend(sync)

        timers("reshape").start(sync)
        logits = input_pool.reshape_logits(logits, unpadded_input, tuple(input_ids.shape))
        # logits = input_pool.reshape_logits_legacy(logits, input_index, tuple(input_ids.shape))
        logits_step = torch.from_numpy(logits).to(torch_device).float()
        timers("reshape").suspend(sync)
        return InferenceFuncOutput(logits_step, kv, None, None)

    inference_func_config = InferenceFuncConfig()
    return WrappedInferenceFunc(inference_func,
                                inference_func_config,
                                executable,
                                transformer_config)

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