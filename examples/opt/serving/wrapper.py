from typing import Sequence

from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass


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

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past
