from collections import namedtuple
import os
import time
from typing import Sequence

import alpa
import numpy as np
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass


from playground.model.opt_model import get_pipeshard_executable, get_config, build_init_cache


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: any = None
    past_key_values: any = None


@dataclass
class InferenceFuncConfig:
    """Implements a minimal config class for using huggingface's generator."""
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
    def __init__(self, inference_func, config):
        self.inference_func = inference_func 
        self.config = config
        self.main_input_name = "input_ids"

    def forward(self, attention_mask):
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
                 past_key_values = None,
                 output_attentions = None,
                 output_hidden_states = None,
                 return_dict = None):
        for i in range(input_ids.shape[1]):
            ret = self.inference_func(input_ids[:,i:i+1],
                                      past_key_values)
            past_key_values = ret.past_key_values
        return ret


def get_model(model_name):
    if "gpt" in model_name:
        raw_model = GPT2LMHeadModel.from_pretrained(model_name)

        def inference_func(input_ids, past_key_values):
            out = raw_model(input_ids=input_ids, past_key_values=past_key_values)
            return InferenceFuncOutput(out.logits, out.past_key_values)

        inference_func_config = raw_model.config

    elif "facebook/opt" in model_name:
        raw_model = OPTForCausalLM.from_pretrained(model_name)

        def inference_func(input_ids, past_key_values):
            if past_key_values is None:
                attention_mask = None
            else:
                past_length = past_key_values[0][0].shape[2]
                attention_mask = torch.ones((input_ids.shape[0], past_length+1))
            out = raw_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
            return InferenceFuncOutput(out.logits, out.past_key_values)

        inference_func_config = InferenceFuncConfig()
        for key in inference_func_config.__dataclass_fields__.keys():
            setattr(inference_func_config, key, getattr(raw_model.config, key))
        print(inference_func_config)

    elif "alpa/opt" in model_name:
        name = model_name.split("-")[1].upper()
        config = get_config(name, num_pp_stages=4)
        # ckpt_dir = os.path.abspath(f"{name}_ts_weights")
        ckpt_dir = "/home/ubuntu/parax-efs/pycharm/opt/alpa_weights/125M/125M_ts_weights"

        alpa.init()
        executable, _ = get_pipeshard_executable(config)
        params_info, _ = executable.get_load_info()
        params = alpa.restore_checkpoint(ckpt_dir, 1, params_info, params_info)

        step_ct = 0

        def inference_func(input_ids, past_key_values):
            nonlocal step_ct

            assert input_ids.shape[1] == 1, f"{input_ids.shape}"
            if past_key_values is None:
                past_key_values = build_init_cache(config)
                step_ct = 0

            input_ids_step = input_ids.numpy()
            position_ids_step = np.full_like(input_ids_step, step_ct + config.pad + 1)

            logits_step, past_key_values = executable(params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": past_key_values,
            })

            step_ct += 1
            return InferenceFuncOutput(torch.from_numpy(np.array(logits_step)), past_key_values)

        inference_func_config = InferenceFuncConfig()

    return WrappedInferenceFunc(inference_func, inference_func_config)


model_name = "alpa/opt-125m"
# "alpa/opt-125m"
# "facebook/opt-125m"
# "gpt2"

torch.manual_seed(8)
prompt = "Computer science is the study of computation and"

tokenizer = GPT2Tokenizer.from_pretrained(model_name.replace("alpa", "facebook"))
model = get_model(model_name)

input_ids = tokenizer(prompt, return_tensors="pt").input_ids
generated_ids = model.generate(input_ids=input_ids, max_length=20, do_sample=True)
generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

print(input_ids)
print(generated_string)
