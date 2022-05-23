import os
from collections import namedtuple
import time
import torch
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass


@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: any = None
    past_key_values: any = None


class WrappedInferenceFunc(GenerationMixin):
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
                 past_key_values=None,
                 output_attentions=None,
                 output_hidden_states=None,
                 return_dict=None):
        for i in range(input_ids.shape[1]):
            ret = self.inference_func(input_ids[:, i:i + 1],
                                      past_key_values)
            past_key_values = ret.past_key_values
        return ret

    # @staticmethod
    # def _reorder_cache(past, beam_idx):
    #     reordered_past = ()
    #     for layer_past in past:
    #         reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
    #     return reordered_past


# model_name = "gpt2"
# model_name = "facebook/opt-2.7b"
# model_name = "facebook/opt-1.3b"

model_name = "facebook/opt-125m"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/1.3B"
# vocab_file = os.path.join(path, "gpt2-vocab.json")
# merges_file = os.path.join(path, "gpt2-merges.txt")
# tokenizer = GPT2Tokenizer(vocab_file, merges_file)

if "gpt" in model_name:
    raw_model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0").half()


    def inference_func(input_ids, past_key_values):
        assert input_ids.shape[1] == 1, f"{input_ids.shape}"
        out = raw_model(input_ids=input_ids,
                        past_key_values=past_key_values)
        return InferenceFuncOutput(out.logits, out.past_key_values)

elif "opt" in model_name:
    raw_model = OPTForCausalLM.from_pretrained(model_name).to("cuda:0").half()


    def inference_func(input_ids, past_key_values):
        assert input_ids.shape[1] == 1, f"{input_ids.shape}"
        if past_key_values is None:
            attention_mask = None
        else:
            past_length = past_key_values[0][0].shape[2]
            attention_mask = torch.ones((input_ids.shape[0], past_length + 1)).to("cuda:0")
        out = raw_model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values)
        return InferenceFuncOutput(out.logits, out.past_key_values)

model = WrappedInferenceFunc(inference_func, raw_model.config)

torch.manual_seed(9)
# prompt = "Computer science is the study of computation and"
# prompt = "Today is a beautiful day and I want to"

prompts = [
   "Today is a beautiful day and I want to",
   "In the city of",
   "Paris is the capital of France and",
   "Computers and mobile phones have taken",
]
# prompt = "Today is a beautiful day and I want to"
#
# def normalize_newlines(s: str):
#     """
#     normalizes new lines, i.e. '\r\n' to '\n'
#     """
#     # note that web browsers send \r\n but our training data uses \n.
#     return s.replace("\r\n", "\n").replace("\r", "\n")
#
# prompt = normalize_newlines(prompt)
# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
for prompt in prompts:
    print(f"Prompt: {prompt}...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # input_ids = torch.cat([torch.zeros([4, 1], dtype=input_ids.dtype), input_ids], dim=-1)
    input_ids = input_ids.to("cuda:0")

    # with torch.no_grad():
    #     logits = raw_model(input_ids)[0]
    #
    # print(logits)
    # pred_next_token = torch.argmax(logits[0, -1], -1)
    # next_token = tokenizer.convert_ids_to_tokens([pred_next_token])
    # next_token = next_token[0].replace("Ġ", "")
    # print(next_token)

    generated_ids = model.generate(input_ids=input_ids, max_length=50, do_sample=True)
    generated_string = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # print(input_ids)
    print(generated_string)


