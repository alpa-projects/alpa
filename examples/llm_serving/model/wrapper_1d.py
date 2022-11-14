import logging
import time
from typing import Union, List

import cupy
import jax
import jax.numpy as jnp
import numpy as np
import os
import torch
import tqdm
from llm_serving.model import opt_model_1d
from transformers import OPTForCausalLM, BloomForCausalLM
from transformers.generation_utils import dataclass

from alpa.timer import timers
from examples.llm_serving.model import opt_model
from examples.llm_serving.model.opt_model_1d import IterationLevelInputPool, unpad, \
    pad
from examples.llm_serving.model.opt_utils import sync
from examples.llm_serving.model.wrapper import disable_torch_init, restore_torch_init

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class InputPoolConfig:
    """The config for iterative-level input pool."""
    batch_size: int = 512
    cache_size: int = 4096


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
        iter = 0
        input_pool.enter_prompts(input_ids)
        while not input_pool.is_finished():
            # tic = time.time()
            input, input_index, position_ids, logit_positions = input_pool.next()
            # timers("enter").suspend(sync)
            batch = {
                "input_ids": input,
                "position_ids": position_ids,
                "cache": input_pool.cache
            }
            # compute
            # timers("compute").start(sync)
            logits = self.executable(self.params, batch)
            # timers("compute").suspend(sync)

            # timers("generate").start(sync)
            if not do_sample:
                generated_ids = self._generate_greedy(logits, logit_positions)
            else:
                raise NotImplementedError()
            # timers("generate").suspend(sync)

            # timers("update").start(sync)
            input_pool.update(generated_ids)
            # timers("update").suspend(sync)
            # elapsed = time.time() - tic
            iter += 1
            # print(f"Iter {iter} takes {elapsed}")

        ret = input_pool.get_results()
        padded_input = np.array(pad(ret))
        latency = input_pool.get_latency()
        return padded_input, latency

    @staticmethod
    def _generate_greedy(logits, positions):
        # outputs = []
        next_token = np.array(jnp.argmax(logits, axis=-1))
        outputs = next_token[positions].tolist()
        # for pos in positions:
        #     outputs.append(int(next_token[pos]))
        return outputs


def get_model(model_name: str,
                path: str,
                dummy: bool = False,
                # batch size, this batch is #tokens
                batch_size: int = 256,
                max_seq_len: int = 2048,
                cache_size: int = 4096,
                # model parameters
                dtype=jnp.float16,
                # Shared arguments with model.generate
                do_sample: bool = False):
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
    executable, params_aval = opt_model_1d.get_jax_executable(model_config)

    # load params
    # TODO(Hao): use the same func with 2D
    params = opt_model_1d.load_params_np(params_aval, path, model_config, dummy)
    params = jax.tree_map(jnp.array, params)

    input_pool_config = InputPoolConfig(batch_size=batch_size,
                                        cache_size=cache_size)

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
