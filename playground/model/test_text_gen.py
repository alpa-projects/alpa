from collections import namedtuple
import os
import time
from typing import Sequence, Any

import alpa
import numpy as np
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass

from opt_model import (get_config, get_pipeshard_executable, load_params_dis_array,
                       init_cache_dis_array)


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: Any = None
    past_key_values: Any = None
    hidden_states: Any = None
    attentions: Any = None


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
                                      past_key_values,
                                      output_hidden_states=output_hidden_states,
                                      output_attentions=output_attentions)
            past_key_values = ret.past_key_values
        return ret


def get_model(model_name, support_output_attentions=False,
              support_output_hidden_states=False):
    if "gpt" in model_name:
        raw_model = GPT2LMHeadModel.from_pretrained(model_name)

        def inference_func(input_ids, past_key_values, output_attentions=False,
                           output_hidden_states=False):
            out = raw_model(input_ids=input_ids,
                            past_key_values=past_key_values,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)
            return InferenceFuncOutput(out.logits, out.past_key_values)

        inference_func_config = raw_model.config

    elif "facebook/opt" in model_name:
        raw_model = OPTForCausalLM.from_pretrained(model_name)

        def inference_func(input_ids, past_key_values,
                           output_attentions=False,
                           output_hidden_states=False):
            if past_key_values is None:
                attention_mask = None
            else:
                past_length = past_key_values[0][0].shape[2]
                attention_mask = torch.ones((input_ids.shape[0], past_length+1))
            out = raw_model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states)
            return InferenceFuncOutput(out.logits, out.past_key_values)

        inference_func_config = InferenceFuncConfig()
        for key in inference_func_config.__dataclass_fields__.keys():
            setattr(inference_func_config, key, getattr(raw_model.config, key))
        print(inference_func_config)

    elif "alpa/opt" in model_name:
        name = model_name.split("-")[1].upper()
        config = get_config(name, num_pp_stages=2)
        path = f"/home/ubuntu/opt_weights/{name}_np"

        alpa.init()
        dummy = False
        executable, params_aval = get_pipeshard_executable(
            config,
            support_output_attentions=support_output_attentions,
            support_output_hidden_states=support_output_hidden_states)
        params = load_params_dis_array(path, executable, params_aval, config, dummy=dummy)
        init_cache = init_cache_dis_array(executable, config, dummy=dummy)

        step_ct = 0

        def inference_func(input_ids, past_key_values, output_attentions=False,
                           output_hidden_states=False):
            nonlocal step_ct

            if past_key_values is None:
                past_key_values = init_cache
                step_ct = 0

            input_ids_step = input_ids.numpy()
            position_ids_step = np.full_like(input_ids_step, step_ct + config.pad + 1)

            output = executable(params, {
                "input_ids": input_ids_step,
                "position_ids": position_ids_step,
                "cache": past_key_values,
            })
            logits_step = output.last_hidden_state
            logits_step = torch.from_numpy(np.array(logits_step))
            past_key_values = output.attention_cache

            step_ct += 1
            return InferenceFuncOutput(logits_step, past_key_values,
                                       output.hidden_states,
                                       output.attentions)

        inference_func_config = InferenceFuncConfig()

    return WrappedInferenceFunc(inference_func, inference_func_config)


tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
model = get_model("alpa/opt-125M")

prompts = [
    "Computer science is the study of computation and",
    "Ion Stoica is a Romanian-American computer scientist specializing in",
    "The University of California, Berkeley is a public",
]

for prompt in prompts:
    tic = time.time()
    torch.manual_seed(8)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = model.generate(input_ids=input_ids, max_length=20, do_sample=True)
    generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    duration = time.time() - tic
    print(f"{generated_string}, speed: {len(generated_ids)/duration:.2f} token/s")
