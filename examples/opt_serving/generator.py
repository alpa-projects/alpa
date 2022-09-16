import time
from typing import List, Optional

import numpy as np
import torch
from transformers import AutoTokenizer

from opt_serving.model.wrapper import get_model
from opt_serving.model.opt_utils import compute_gpt_tflops_inference_with_padding
from opt_serving.service.constants import MAX_SEQ_LEN, MAX_BS
from opt_serving.service.utils import build_logger


logger = build_logger()


class Generator:
    """Alpa generator interface.

    This class wraps tokenizer and the langauge model.
    """

    def __init__(self,
                 model_name,
                 path,
                 torch_device="cpu",
                 tokenizer_name=None,
                 add_bos_token=False,
                 do_sample=False,
                 num_beams=1,
                 num_return_sequences=1):

        # Model arguments
        self.model_name = model_name
        self.path = path
        self.model_wrapper = None
        self.torch_device = torch_device

        # Tokenizer arguments
        self.tokenizer_name = "facebook/opt-30b" if not tokenizer_name else tokenizer_name
        self.tokenizer = None
        self.add_bos_token = add_bos_token

        # Generation arguments
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences

        # Others
        self.num_gpus = None
        self.dataset_to_epoch_iter = dict()

        # Initialize models
        self.load_model()

    def load_model(self):
        """Compile and load a model."""
        tic = time.time()

        # Init model
        self.model_wrapper = get_model(self.model_name, self.path,
                                       torch_device=self.torch_device,
                                       autoregressive=True,
                                       batch_size=MAX_BS,
                                       encoder_chunk_sizes=[1, 64],
                                       max_target_positions=MAX_SEQ_LEN,
                                       num_beams=self.num_beams,
                                       num_return_sequences=self.num_return_sequences,
                                       do_sample=self.do_sample)
        load_time = time.time() - tic

        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.tokenizer.add_bos_token = False

        if "alpa" in self.model_name:
            import alpa
            self.num_gpus = alpa.get_global_cluster().num_devices
        else:
            self.num_gpus = 1
        logger.info(f"Loading model time: {load_time:.2f}")

    def encode(self, s: str):
        """Tokenize strings"""
        # note that web browsers send \r\n but our training data uses \n.
        s = s.replace("\r\n", "\n").replace("\r", "\n")
        return self.tokenizer.encode(s)

    def generate(
        self,
        inputs: List[List[int]],
        min_tokens: List[int],
        max_tokens: List[int],
        temperature: float,
        top_p: float,
        n: int,
        echo: bool,
        best_of: int,
    ):
        """
        Generation API.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create

        Args:
          inputs: a list of tokenized inputs.
          min_tokens: The minimum number of tokens to generate.
          max_tokens: The maximum number of tokens to generate.
          temperature: What sampling temperature to use.
          top_p: The nucleus sampling probability.
          n: How many completions to generate for each prompt.
          echo: if true, returned text/tokens/scores includes the prompt.
          best_of: Generates best_of completions server-side and returns the "best" (the one with the highest log probability per token)
        """
        start_time = time.time()
        total_inference_time = 0
        batch_request_uuid = next_serve_batch_uuid()
        ori_bs = len(inputs)
        logger.info(f"Batch {batch_request_uuid} begin. batch size: {ori_bs}")

        # Check arguments
        assert best_of == self.num_beams, "model must be instantiated and used with the same num_beams"
        assert n == self.num_return_sequences, "model must be instantiated and used with the same num_return_sequences"
        if temperature <= 1e-3:
            do_sample = False
        else:
            do_sample = self.do_sample
        # Resolve the max sequence length allowed from multiple sources
        max_seq_len = min(MAX_SEQ_LEN, self.model_wrapper.transformer_config.seq_len)

        # Pad the batch to a maximum batch size
        input_ids = pad_batch(inputs, self.tokenizer.pad_token_id, MAX_BS)
        input_ids = torch.IntTensor(input_ids).to(self.torch_device)
        input_lens = [len(x) for x in inputs]
        batch_size = len(input_ids)

        # Set generation args
        if min_tokens is None:
            min_tokens = [0] * batchsize
        if max_tokens is None:
            max_tokens = [max_seq_len] * batchsize
        min_length = max(min_tokens) + max(input_lens)
        max_length = min(max_seq_len, max(max_tokens) + max(input_lens))

        generator_args = {
            "min_length": min_length,
            "max_length": max_length,
            "temperature": temperature,
            "do_sample": do_sample,
            "top_p": top_p,
            "num_beams": best_of,
            "num_return_sequences": n,
            "early_stopping": True,
            "repetition_penalty": 1.0,
            "no_repeat_ngram_size": 8,
        }

        logger.info(
            f"Generate begin. batch uuid: {batch_request_uuid}, "
            f"batch_size: {batch_size}, original bs: {ori_bs}, "
            f"generator_args: {generator_args}.")

        inference_start_time = time.time()
        output_ids = self.model_wrapper.generate(input_ids=input_ids, **generator_args)
        inference_time = time.time() - inference_start_time
        output_ids = torch.reshape(output_ids, (batch_size, self.num_return_sequences, -1))

        tflops, speed, token_32_latency = self.estimate_performance(
            output_ids, inference_time)

        logger.info(f"Generate end. batch uuid: {batch_request_uuid}")

        # Decode results to strings
        ret = []
        for i in range(ori_bs):
            tmp_ret = []
            for tokens in output_ids[i]:
                prompt_len = input_lens[i]
                if echo:
                    tokens = tokens[:prompt_len + max_tokens[i]]
                else:
                    tokens = tokens[prompt_len:prompt_len + max_tokens[i]]
                text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                result = {"text": text}
                tmp_ret.append(result)
            ret.append(tmp_ret)

        logger.info(f"Batch {batch_request_uuid} end. batch size: {ori_bs}, "
                    f"e2e latency: {time.time() - start_time:.2f} s, "
                    f"inference latency: {inference_time:.2f} s, "
                    f"speed: {speed:.2f} token/s, "
                    f"32 token latency: {token_32_latency:.2f} s, "
                    f"tflops: {tflops:.2f} TFLOPS")
        return ret

    def forward(
        self,
        inputs,
        cache_id,
        pasts=None,
    ):
        logger.info(f"Forward begin. cache_id: {cache_id}")
        time_start = time.time()

        inputs = pad_batch(inputs, self.tokenizer.pad_token_id, MAX_BS)
        input_ids = torch.IntTensor(inputs).to(self.torch_device)

        attention_mask = self.model_wrapper._prepare_attention_mask_for_generation(input_ids, pad_token_id=self.model_wrapper.config.pad_token_id, eos_token_id=self.model_wrapper.config.eos_token_id)
        model_inputs = self.model_wrapper.prepare_inputs_for_generation(input_ids, past=pasts[cache_id][1] if pasts is not None else None, attention_mask=attention_mask)
        output = self.model_wrapper(**model_inputs)

        logger.info(f"Forward end. e2e latency: {time.time() - time_start:.2f}")
        return output

    def estimate_performance(self, output_ids, latency):
        """Report the tflops, decoding speed, and latency for decoding 32 tokens."""
        # TODO(Hao): (1) we are still over-computing
        transformer_config = self.model_wrapper.transformer_config

        batch_size = self.num_beams * len(output_ids)
        gen_len = max(t[0].shape[0] for t in output_ids)
        seq_len = transformer_config.seq_len
        H = transformer_config.H
        L = transformer_config.L
        vocab_size = transformer_config.vocab_size
        tflops = compute_gpt_tflops_inference_with_padding(
            batch_size, gen_len, seq_len, L, H, vocab_size, self.num_gpus,
            latency)
        speed = batch_size * gen_len / latency
        token_32_latency = 32.0 / (speed / len(output_ids))
        return tflops, speed, token_32_latency


def pad_batch(inputs, pad_value, max_batch_size):
    """Pad the batch to max_batch_size."""
    new_inputs = inputs
    src_lens = [len(input) for input in inputs]
    max_len = max(src_lens)
    bs = len(inputs)

    # Pad to max_len
    for new_input in new_inputs:
        ori_len = len(new_input)
        if len(new_input) < max_len:
            new_input.extend([pad_value for _ in range(max_len - ori_len)])

    # Pad to max_batch_size
    if bs < max_batch_size:
        new_inputs.extend([[pad_value for _ in range(max_len)] for _ in range(MAX_BS - bs)])
    return new_inputs


serve_batch_counter = 0

def next_serve_batch_uuid(number=1):
    """Return the next uuid of a remote buffer."""
    global serve_batch_counter
    if number == 1:
        ret = serve_batch_counter
    else:
        ret = np.arange(serve_batch_counter, serve_batch_counter + number)
    serve_batch_counter = (serve_batch_counter + number) % (1 << 60)
    return ret
