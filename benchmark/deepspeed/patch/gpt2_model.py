# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch

from megatron import get_args
from megatron import mpu
from megatron.module import MegatronModule

from .language_model import parallel_lm_logits
from .language_model import get_language_model
from .utils import init_method_normal
from .utils import scaled_init_method_normal

import deepspeed

def gpt2_attention_mask_func(attention_scores, ltor_mask):
    attention_scores.masked_fill_(ltor_mask, -10000.0)
    return attention_scores


class GPT2Model(MegatronModule):
    """GPT-2 Language model."""

    def __init__(self, num_tokentypes=0, parallel_output=True):
        super(GPT2Model, self).__init__()
        args = get_args()

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        self.language_model, self._language_model_key = get_language_model(
            attention_mask_func=gpt2_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std,
                                                         args.num_layers))


    def forward(self, input_ids, position_ids, attention_mask, labels=None,
                tokentype_ids=None, layer_past=None, get_key_value=False,
                forward_method_parallel_output=None, curriculum_seqlen=None):
        if curriculum_seqlen is not None:
            args = get_args()
            args.curriculum_seqlen = curriculum_seqlen
            if curriculum_seqlen < input_ids.size()[1]:
                # seqlen-based curriculum learning
                # input_ids, position_ids, labels have size [batch size, seqlen]
                input_ids = input_ids[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                labels = labels[:, :curriculum_seqlen].contiguous()

                # attention_mask has size [1, 1, seqlen, seqlen]
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()

        # Language model.
        lm_output = self.language_model(input_ids,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)
        if get_key_value:
            lm_output, presents = lm_output

        # Output.
        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output

        output = parallel_lm_logits(
            lm_output,
            self.language_model.embedding.word_embeddings.weight,
            parallel_output)

        if get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = mpu.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)
            return loss


    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):

        state_dict_ = {}
        state_dict_[self._language_model_key] \
            = self.language_model.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if self._language_model_key in state_dict:
            state_dict = state_dict[self._language_model_key]
        self.language_model.load_state_dict(state_dict, strict=strict)
