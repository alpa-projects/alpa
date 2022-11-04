import logging
from typing import Union, List

import numpy as np
import os

import jax
import jax.numpy as jnp
import torch
import tqdm
from llm_serving.model import opt_model_1d
import cupy

from alpa.timer import timers
from transformers import OPTForCausalLM, BloomForCausalLM
from transformers.generation_utils import dataclass
from examples.llm_serving.model.opt_utils import TransformerModelConfig
from examples.llm_serving.model.wrapper import disable_torch_init, restore_torch_init, InferenceFuncOutput, \
    InferenceFuncConfig, WrappedInferenceFunc
from examples.llm_serving.model import opt_model
from examples.llm_serving.model.opt_model_1d import IterationLevelInputPool, unpad, pad, custom_reshape_logits
from examples.llm_serving.model.opt_utils import sync
from alpa.collective.worker_nccl_util_cupy import jax_tensor_to_cupy


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class InputPoolConfig:
    """The config for iterative-level input pool."""
    batch_size: int = 512
    cache_size: int = 4096
    max_cache_per_seq: int = 128


class SequenceGenerator:
    def __init__(self, executable, params, input_pool_config, model_config):
        self.executable = executable
        self.params = params
        self.input_pool_config = input_pool_config
        self.model_config = model_config
        # some other attributes
        self.pad = self.model_config.pad

    def generate(self,
                 input: Union[IterationLevelInputPool, List[List[int]], np.ndarray],
                 max_length=None,
                 max_new_tokens=None,
                 do_sample=False,
                 **kwargs):
        if max_length == None and max_new_tokens == None:
            raise RuntimeError("Please provide at least one of max_length and max_new_tokens.")

        if isinstance(input, IterationLevelInputPool):
            raise NotImplementedError()
        elif isinstance(input, (List, np.ndarray, torch.Tensor)):
            unpadded_input = unpad(input)
            return self.generate_by_batch(unpadded_input,
                                          max_length=max_length,
                                          max_new_tokens=max_new_tokens,
                                          do_sample=do_sample)
        else:
            raise RuntimeError()

    def generate_by_batch(self,
                          input_ids: List[List[int]],
                          max_length=None,
                          max_new_tokens=None,
                          do_sample=False):
        input_pool = IterationLevelInputPool(self.input_pool_config,
                                             self.model_config,
                                             max_length=max_length,
                                             max_new_tokens=max_new_tokens)
        input_pool.enter_prompts(input_ids)
        iteration = 0
        while not input_pool.is_finished():
            timers("enter").start(sync)
            input, input_index, position_ids, logit_positions = input_pool.next()
            timers("enter").suspend(sync)
            # print(input_pool.cache.kv_caches_cupy[0][0][0:512, 0, 0])
            os.environ["FT_INPUT_INDEX_ADDR"] = str(input_index.ctypes.data)
            os.environ["FT_CACHE_INDEX_ADDR"] = str(input_pool.cache.kv_cache_ids.ctypes.data)
            os.environ["FT_MAX_CACHE_LEN_PER_SEQ"] = str(input_pool.max_cache_per_seq)
            batch = {
                "input_ids": input,
                "position_ids": position_ids,
                "cache": input_pool.cache.kv_caches
            }
            # compute
            timers("compute").start(sync)
            logits, kv = self.executable(self.params, batch)
            timers("compute").suspend(sync)

            timers("generate").start(sync)
            if not do_sample:
                generated_ids = self._generate_greedy(logits, logit_positions)
            else:
                raise NotImplementedError()
            timers("generate").suspend(sync)

            timers("update").start(sync)
            input_pool.update_cache(kv, generated_ids)
            timers("update").suspend(sync)
            iteration += 1

        ret = input_pool.get_results()
        padded_input = np.array(pad(ret))
        return padded_input

    @staticmethod
    def _generate_greedy(logits, positions):
        # outputs = []
        next_token = np.array(jnp.argmax(logits, axis=-1))
        outputs = next_token[positions].tolist()
        # for pos in positions:
        #     outputs.append(int(next_token[pos]))
        return outputs

    @staticmethod
    def _generate_greedy_v2(logits, positions):
        src_indices = cupy.array(positions)
        src_logits =  jax_tensor_to_cupy(logits)
        dst_shape = (len(positions), 50272)
        dst_logits = cupy.zeros(dst_shape, dtype=logits.dtype)
        num_blocks = len(positions)
        custom_reshape_logits[num_blocks, 1024](dst_logits.ravel(), src_logits.ravel(), src_indices)
        next_token = cupy.asnumpy(cupy.argmax(dst_logits, axis=-1)).tolist()
        return next_token


def get_model(model_name: str,
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
                # num_beams: int = 1,
                # num_return_sequences: int = 1,
                # return_dict_in_generate: bool = True,
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
    model_config = opt_model.get_config(name, dtype=dtype, max_seq_len=max_seq_len)
    # transformer_config = TransformerModelConfig(
    #     H=config.hidden_size,
    #     L=config.num_hidden_layers,
    #     n_head=config.n_head,
    #     seq_len=config.max_seq_len,
    #     vocab_size=config.vocab_size)
    executable, params_aval = opt_model_1d.get_jax_executable(
        model_config,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states)

    # load params
    # TODO(Hao): use the same func with 2D
    params = opt_model_1d.load_params_np(params_aval, path, model_config, dummy)
    params = jax.tree_map(jnp.array, params)

    input_pool_config = InputPoolConfig(batch_size=batch_size,
                                        cache_size=cache_size,
                                        max_cache_per_seq=max_cache_per_seq)

    return SequenceGenerator(executable, params, input_pool_config, model_config)


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
