import logging
import time
from typing import List, Optional, Dict

import numpy as np
import torch
from torch import Tensor

from examples.opt.serving import utils
from metaseq import tasks
from examples.opt.model.test_text_gen import get_model
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel

logger = logging.getLogger(__name__)


class GeneratorInterface:
    """Alpa generator interface"""
    def __init__(self, cfg, dummy=False):
        self.cfg = cfg

        self.model_wrapper = None
        self.task = None
        self.tokenizer = None

        self.dummy = dummy


    def load_model(self, model_name="alpa/opt-125m"):
        """Load model and return the model wrapper."""
        self.task = tasks.setup_task(self.cfg.task)
        self.model_wrapper = get_model(model_name, "cuda", False)
        self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")
        return self.model_wrapper

    def generate(
            self,
            inputs: List[List[int]],
            min_tokens: List[int] = None,
            max_tokens: List[int] = None,
            temperature: float = 1.0,
            top_p: float = -1.0,
            logprobs: int = 0,
            n: int = 1,
            best_of: Optional[int] = None,
            echo: bool = False,
            stop: Optional[List[int]] = None,
            seed: Optional[int] = None,
            use_cuda: bool = True,
    ):
        """
        Generate from sequences.
        Parameters match those of the OpenAI API.
        https://beta.openai.com/docs/api-reference/completions/create
        inputs: a list of pre-tokenized prompts
        min_tokens: blocks EOS until at least this many tokens is provided
        max_tokens: forces EOS after this many tokens
        temperature: softmax temperature
        top_p: nucleus probability
        log_probs: return this cutoff of the probability distribution
        n: beam size
        best_of: number of beams to return. must be <= n
        echo: if true, returned text/tokens/scores includes the prompt.
            This is useful for getting PPL evaluations.
        stop: a list of terminating tokens
        seed: an integer if desired
        use_cuda: should we use GPUs.
        """
        # if seed:
        #     utils.set_torch_seed(seed)
        if self.tokenizer is None or self.model_wrapper is None:
            raise RuntimeError("Do you call load_model()?")

        start_time = time.time()
        total_generation_time = 0

        # Generator args
        if not best_of:
            best_of = n
        assert best_of >= n
        beam = best_of
        sampling_topp = top_p if top_p > 0 else -1
        sampling = top_p > 0.0
        temperature = temperature if temperature > 0 else 1.0

        # MAX_SEQ_LEN = utils.resolve_max_positions(
        #     self.task.max_positions(), *[model.max_positions() for model in [model]]
        # )
        MAX_SEQ_LEN = utils.resolve_max_positions(self.task.max_positions(), 2048)

        # TODO(roller): simplify
        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        batches = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
        ).next_epoch_itr(shuffle=False)
        for batch in batches:
            src_tokens = batch["net_input"]["src_tokens"]
            src_lengths = batch["net_input"]["src_lengths"]
            batchsize = src_tokens.size(0)

            # set generation args
            # prevent us from ever generating past our max sequence length
            if max_tokens is None:
                max_tokens = [MAX_SEQ_LEN] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(
                MAX_SEQ_LEN, max(max_tokens) + src_lengths.max().item()
            )
            total_min_tokens = max(min_tokens) + src_lengths.max().item()
            min_len = total_min_tokens
            max_len = total_max_tokens

            # generator = self.task.build_generator(
            #     self.models, self.cfg.generation, extra_gen_cls_kwargs={"stop": stop}
            # )
            generator_args = {
                "beam_size": beam,
                "max_len": max_len,
                "min_len": min_len,
                "temperature": temperature,
                "sampling": sampling,
                "top_p": sampling_topp,
            }
            logger.info(f"Preparing generator with settings {generator_args}")
            generator = Generator(self.model_wrapper, **generator_args)

            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            translate_start_time = time.time()
            translations = generator.generate(batch["net_input"]["src_tokens"])
            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            # possibly cut off any bsz padding we did
            translations = translations[: len(inputs)]
            # actually turn everything into strings
            for i in range(len(translations)):
                decoding = translations[i]
                beams = []
                for beam in decoding:
                    # first beam is always the highest scoring
                    tokens = beam["tokens"].tolist()  # implicit move to cpu
                    scores = beam["positional_scores"].tolist()
                    if logprobs > 0:
                        distributions = beam["distributions"].cpu()
                    else:
                        distributions = None

                    tokens, scores, distributions = GeneratorInterface._filter_special(
                        tokens, scores, distributions
                    )
                    prompt_len = src_lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[: prompt_len + max_tokens[i] - 1]
                        scores = scores[: prompt_len + max_tokens[i] - 1]
                        if logprobs > 0:
                            distributions = distributions[
                                            : prompt_len + max_tokens[i] - 1
                                            ]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len - 1:][: max_tokens[i]]
                        scores = scores[prompt_len - 1:][: max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[prompt_len - 1:][
                                            : max_tokens[i]
                                            ]
                    # turn it into a string
                    text = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    result = {
                        "text": text,
                        "tokens": [self.tokenizer.decode([t]) for t in tokens],
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        "text_offset": None,
                        "token_scores": scores,
                    }
                    if logprobs > 0:
                        # final result is a List[Dict[str, float]]
                        # where each item in the list corresponds to a token in the
                        # sequence, and the dict provides the probabilities of the
                        # top-k tokens at that timestep.
                        out_logprobs = []
                        all_top_toks, all_top_scores = distributions.topk(
                            k=logprobs, dim=-1
                        )
                        for top_scores, top_toks in zip(all_top_toks, all_top_scores):
                            lp = {
                                self.bpe.bpe.decode([t.item()]): s.item()
                                for t, s in zip(top_toks, top_scores)
                            }
                            out_logprobs.append(lp)
                        result["top_logprobs"] = out_logprobs
                    else:
                        result["top_logprobs"] = None

                    beams.append(result)
                retval.append(beams)

        logger.info(
            "Total time: {:.3f} seconds; generation time: {:.3f}".format(
                time.time() - start_time, total_generation_time
            )
        )
        return retval

    @staticmethod
    def _filter_special(
            tokens: List[int],
            scores: List[float],
            distributions,
            pad_token: int = 1,
    ):
        """
        Cut off tokens after finding a special tokens.
        """

        # tokens is a 1D list of token IDs of length seqlen
        # scores is a 1D list of log-probability scores for those tokens (length seqlen)
        # distributions (optional) is a seqlen x vocab_size tensor corresponding to
        # the full distribution of predictions at each timestep

        output = []
        mask = []
        for t, s in zip(tokens, scores):
            if t == pad_token:
                # simply skip pads
                mask.append(False)
                continue
            if t <= 3:
                # and other special tokens should end things
                break
            mask.append(True)
            output.append((t, s))
        new_tokens, new_scores = zip(*output)

        # cut off at stop and drop pads
        if distributions is not None:
            distributions = distributions[: len(mask)][mask]
            distributions = distributions[: len(output)]
        return new_tokens, new_scores, distributions


class Generator:
    def __init__(self,
                 model_wrapper,
                 beam_size: int = 1,
                 max_len: int = 200,
                 min_len: int = 1,
                 temperature: float = 1.0,
                 sampling=False,
                 top_k=0.0,
                 top_p=0.0):
        """Generator."""
        self.model_wrapper = model_wrapper
        self.vocab_size = 50272
        # Params copied from Metaseq/SequenceGenerator
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len = max_len
        self.min_len = min_len

        self.sampling = sampling
        self.top_k = top_k
        self.top_p = top_p

        self.temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"

    def generate(self, input_ids):
        output = self.model_wrapper.generate(
            input_ids=input_ids,
            min_length=self.min_len,
            max_length=self.max_len,
            temperature=self.temperature,
            do_sample=self.sampling,
            # top_k=self.top_k,
            top_p=self.top_p,
            early_stopping=True,
            no_repeat_ngram_size=2
            # return_dict_in_generate=True
            # output_hidden_states=True
        )
        generated_ids = output
        retvals = [[{} for _ in range(self.beam_size)] for _ in generated_ids]
        for g in range(generated_ids.shape[0]):
            for beam in range(self.beam_size):
                retvals[g][beam] = {"tokens": generated_ids[g, 1:],
                                    "positional_scores": torch.zeros_like(generated_ids[g, 1:], dtype=torch.float16)}
        return retvals
