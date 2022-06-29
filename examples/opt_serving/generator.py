import numpy as np
import time
from typing import List, Optional

import torch
from transformers import AutoTokenizer

from examples.opt_serving.model.wrapper import get_model
from examples.opt_serving.dataset import TokenBlockDataset, PadDataset, NestedDictionaryDataset, NumelDataset, \
    BaseDataset, iterators
from examples.opt_serving.service.utils import build_logger
from examples.opt_serving.dataset import data_utils
from examples.opt_serving.dataset.prepend_token_dataset import PrependTokenDataset
from examples.opt_serving.dataset.strip_token_dataset import StripTokenDataset
from examples.opt_serving.service.constants import MAX_SEQ_LEN
from examples.opt_serving.model.opt_utils import compute_gpt_tflops_inference_with_padding

logger = build_logger()


def apply_to_sample(f, sample):
    if hasattr(sample, "__len__") and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return tuple(_apply(x) for x in x)
        elif isinstance(x, set):
            return {_apply(x) for x in x}
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, device=None):
    device = device or torch.cuda.current_device()

    def _move_to_cuda(tensor):
        # non_blocking is ignored if tensor is not pinned, so we can always set
        # to True (see github.com/PyTorchLightning/pytorch-lightning/issues/620)
        return tensor.to(device=device, non_blocking=True)

    return apply_to_sample(_move_to_cuda, sample)


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


def filter_indices_by_size(indices,
                           dataset,
                           max_positions=None,
                           ignore_invalid_inputs=False):
    """Filter examples that are too large.

    Args:
        indices (np.array): original array of sample indices
        dataset (~metaseq.data.BaseDataset): dataset to batch
        max_positions (optional): max sentence length supported by the
            model (default: None).
        ignore_invalid_inputs (bool, optional): don't raise Exception for
            sentences that are too long (default: False).
    Returns:
        np.array: array of filtered sample indices
    """
    indices, ignored = dataset.filter_indices_by_size(indices, max_positions)
    if len(ignored) > 0:
        if not ignore_invalid_inputs:
            raise Exception(
                ("Size of sample #{} is invalid (={}) since max_positions={}, "
                 "skip this example with --skip-invalid-size-inputs-valid-test"
                ).format(ignored[0], dataset.size(ignored[0]), max_positions))
        logger.warning(("{:,} samples have invalid sizes and will be skipped, "
                        "max_positions={}, first few sample ids={}").format(
                            len(ignored), max_positions, ignored[:10]))
    return indices


class GeneratorInterface:
    """Alpa generator interface."""

    def __init__(self,
                 model_name="alpa/opt-125m",
                 path="/home/ubuntu/opt_weights/",
                 tokenizer_name=None,
                 add_bos_token=False):

        self.model_name = model_name
        self.path = path
        self.tokenizer_name = "facebook/opt-30b" if not tokenizer_name else tokenizer_name
        self.add_bos_token = add_bos_token

        self.model_wrapper = None
        self.task = None
        self.tokenizer = None
        self.num_gpus = 1
        self.dataset_to_epoch_iter = dict()

        self.load_model()

    def load_model(self):
        """Load model and return the model wrapper."""
        tic = time.time()
        self.model_wrapper = get_model(self.model_name, "cuda", self.path, True)
        load_time = time.time() - tic

        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        # Disable default add_bos_token behavior and decide if to add it later.
        self.tokenizer.add_bos_token = False
        if "alpa" in self.model_name:
            import alpa
            self.num_gpus = alpa.get_global_cluster().num_devices
        logger.info(f"Loading model time: {load_time:.2f}")

    def legacy_generate(
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

        # Resolve max sequence length from multiple sources.
        max_seq_len = self.max_sequence_len()

        # TODO(roller): simplify
        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        batches = self.task.get_batch_iterator(
            dataset=self.task.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=1,
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
                max_tokens = [max_seq_len] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(max_seq_len,
                                   max(max_tokens) + src_lengths.max().item())
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
            logger.info(
                f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = move_to_cuda(batch)

            translate_start_time = time.time()
            translations = generator.generate(batch["net_input"]["src_tokens"])
            translate_time = time.time() - translate_start_time
            total_generation_time += translate_time

            # possibly cut off any bsz padding we did
            translations = translations[:len(inputs)]
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
                        tokens, scores, distributions)
                    prompt_len = src_lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[:prompt_len + max_tokens[i] - 1]
                        scores = scores[:prompt_len + max_tokens[i] - 1]
                        if logprobs > 0:
                            distributions = distributions[:prompt_len +
                                                          max_tokens[i] - 1]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len - 1:][:max_tokens[i]]
                        scores = scores[prompt_len - 1:][:max_tokens[i]]
                        if logprobs > 0:
                            distributions = distributions[prompt_len -
                                                          1:][:max_tokens[i]]
                    # turn it into a string
                    text = self.tokenizer.decode(tokens,
                                                 skip_special_tokens=True)
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
                            k=logprobs, dim=-1)
                        for top_scores, top_toks in zip(all_top_toks,
                                                        all_top_scores):
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
                time.time() - start_time, total_generation_time))
        return retval

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
        batch_request_uuid = next_serve_batch_uuid()
        if self.tokenizer is None or self.model_wrapper is None:
            raise RuntimeError("Model is not loaded.")

        if logprobs > 0:
            # TODO(Hao): support this
            raise NotImplementedError(
                "logprob>0 is not supported at this moment.")

        start_time = time.time()
        total_inference_time = 0

        # Generator args
        if not best_of:
            best_of = n
        assert best_of >= n
        beam_size = best_of

        # TODO (Hao & Yonghao): support beam search
        if beam_size > 1:
            raise NotImplementedError("We only support beam = 1 now.")

        sampling_topp = top_p if top_p > 0 else -1
        sampling = top_p > 0.0
        temperature = temperature if temperature > 0 else 1.0

        # Resolve max sequence length from multiple sources.
        max_seq_len = self.max_sequence_len()

        retval = []
        tokens = [torch.LongTensor(t) for t in inputs]
        lengths = [len(t) for t in inputs]
        batches = self.get_batch_iterator(
            dataset=self.build_dataset_for_inference(tokens, lengths),
            max_tokens=None,
            max_sentences=1,
            max_positions=None,
            ignore_invalid_inputs=False,
        ).next_epoch_itr(shuffle=False)
        logger.info(f"Serve batch {batch_request_uuid} with {len(batches)} compute batches.")
        for batch_idx, batch in enumerate(batches):
            src_tokens = batch["src_tokens"]
            src_lengths = batch["src_lengths"]
            batchsize = src_tokens.size(0)

            # set generation args
            # prevent us from ever generating past our max sequence length
            if max_tokens is None:
                max_tokens = [max_seq_len] * batchsize
            if min_tokens is None:
                min_tokens = [0] * batchsize
            total_max_tokens = min(max_seq_len,
                                   max(max_tokens) + src_lengths.max().item())
            total_min_tokens = max(min_tokens) + src_lengths.max().item()
            min_len = total_min_tokens
            max_len = total_max_tokens

            generator_args = {
                "beam_size": beam_size,
                "max_len": max_len,
                "min_len": min_len,
                "temperature": temperature,
                "sampling": sampling,
                "top_p": sampling_topp,
            }
            logger.debug(generator_args)
            generator = Generator(self.model_wrapper, **generator_args)

            # okay actually generate
            if use_cuda:
                batch = move_to_cuda(batch)

            inference_start_time = time.time()
            translations = generator.generate(batch["src_tokens"])
            inference_time = time.time() - inference_start_time
            flops, speed, token_32_latency = self.estimate_performance(
                translations, inference_time)
            logger.info(
                "- Serve batch {} / compute batch {} | #batch_size: {}, max_len: {} shape: {}, args: {}, "
                "batch latency (s): {:.2f}, flops: {:.4f}, speed: {:.4f}, 32-token latency: {:.2f}"
                .format(batch_request_uuid, batch_idx, batchsize,
                        max(src_lengths), src_lengths, generator_args,
                        inference_time, flops, speed, token_32_latency))
            total_inference_time += inference_time

            # possibly cut off any bsz padding we did
            translations = translations[:len(inputs)]
            # actually turn everything into strings
            for i in range(len(translations)):
                decoding = translations[i]
                beams = []
                for beam in decoding:
                    # first beam is always the highest scoring
                    tokens = beam["tokens"].tolist()  # implicit move to cpu
                    # scores = beam["positional_scores"].tolist()

                    # tokens, scores, distributions = GeneratorInterface._filter_special(
                    #     tokens, scores, distributions
                    # )
                    prompt_len = src_lengths[i]
                    if echo:
                        # don't cut off prompt
                        tokens = tokens[:prompt_len + max_tokens[i] - 1]
                        # scores = scores[: prompt_len + max_tokens[i] - 1]
                    else:
                        # cut off prompt
                        tokens = tokens[prompt_len - 1:][:max_tokens[i]]
                        # scores = scores[prompt_len - 1:][: max_tokens[i]]
                    # turn it into a string
                    text = self.tokenizer.decode(tokens,
                                                 skip_special_tokens=True)
                    result = {
                        "text": text,
                        # "tokens": [self.tokenizer.decode([t]) for t in tokens],
                        # text offset is useful for cutting off prompts or prefixes
                        # or evaluating PPL on just a subset of tokens
                        # "text_offset": None,
                        # "token_scores": scores,
                    }
                    beams.append(result)
                retval.append(beams)
        logger.info(
            "Serve batch {} completed!  | #samples: {},  #batches: {}, e2e latency (s): {:.2f}, inference (s): {:.2f}"
            .format(batch_request_uuid, len(lengths), len(batches),
                    time.time() - start_time, total_inference_time))
        return retval

    def estimate_performance(self, translations, latency):
        """Report the tflops, decoding speed, and latency for decoding 32 tokens."""
        # TODO(Hao): (1) we are still over-computing (2) support beam > 1
        transformer_config = self.model_wrapper.transformer_config

        beam_size = len(translations[0])
        assert beam_size == 1, "we do not support beam > 1 now."
        batch_size = beam_size * len(translations)
        gen_len = max(t[0]["tokens"].shape[0] for t in translations)
        seq_len = transformer_config.seq_len
        H = transformer_config.H
        L = transformer_config.L
        vocab_size = transformer_config.vocab_size
        flops = compute_gpt_tflops_inference_with_padding(
            batch_size, gen_len, seq_len, L, H, vocab_size, self.num_gpus,
            latency)
        speed = batch_size * gen_len / latency
        token_32_latency = 32.0 / (speed / len(translations))
        return flops, speed, token_32_latency

    # @staticmethod
    # def _filter_special(
    #         tokens: List[int],
    #         scores: List[float],
    #         distributions,
    #         pad_token: int = 1,
    # ):
    #     """
    #     Cut off tokens after finding a special tokens.
    #     """
    #
    #     # tokens is a 1D list of token IDs of length seqlen
    #     # scores is a 1D list of log-probability scores for those tokens (length seqlen)
    #     # distributions (optional) is a seqlen x vocab_size tensor corresponding to
    #     # the full distribution of predictions at each timestep
    #
    #     output = []
    #     mask = []
    #     for t, s in zip(tokens, scores):
    #         if t == pad_token:
    #             # simply skip pads
    #             mask.append(False)
    #             continue
    #         if t <= 3:
    #             # and other special tokens should end things
    #             break
    #         mask.append(True)
    #         output.append((t, s))
    #     new_tokens, new_scores = zip(*output)
    #
    #     # cut off at stop and drop pads
    #     if distributions is not None:
    #         distributions = distributions[: len(mask)][mask]
    #         distributions = distributions[: len(output)]
    #     return new_tokens, new_scores, distributions

    def max_sequence_len(self):
        """Resolve the max sequence length allowed from multiple sources."""
        return min(MAX_SEQ_LEN, self.model_wrapper.transformer_config.seq_len)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """Build a batched dataset for inference"""

        # TODO(Hao): understand TokenBlockDataset and simplify it
        dataset = TokenBlockDataset(src_tokens,
                                    src_lengths,
                                    block_size=None,
                                    pad=self.tokenizer.pad_token_id,
                                    eos=self.tokenizer.eos_token_id,
                                    break_mode="eos")

        # Strip end tokens if there are any
        dataset = StripTokenDataset(dataset, self.tokenizer.eos_token_id)
        # Prepend an EOS; don't ask why
        token_to_prepend = self.tokenizer.bos_token_id if self.add_bos_token else self.tokenizer.eos_token_id
        dataset = PrependTokenDataset(dataset, token=token_to_prepend)
        # Pad when there are various length in a batch
        dataset = PadDataset(dataset,
                             pad_idx=self.tokenizer.pad_token_id,
                             left_pad=False)
        numel_dataset = NumelDataset(dataset, reduce=False)
        dataset = NestedDictionaryDataset(
            {
                "src_tokens": dataset,
                "src_lengths": numel_dataset
            },
            sizes=[np.array(src_lengths)],
        )
        return dataset

    def get_batch_iterator(
        self,
        dataset: BaseDataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        batch_by_size=True,
        skip_remainder_batch=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~metaseq.data.BaseDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator
                (default: False).
            batch_by_size (bool, optional):
                batch sequences of similar length together to reduce padding.
                If false, each batch will be of size max_sentences.
            skip_remainder_batch (bool, optional): if set, discard the last
                batch in each training epoch, as the last batch is often smaller than
                    local_batch_size * distributed_word_size (default: ``True``).
        Returns:
            ~metaseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        if not disable_iterator_cache and dataset in self.dataset_to_epoch_iter:
            logger.debug(
                "reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, BaseDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = filter_indices_by_size(indices, dataset, max_positions,
                                             ignore_invalid_inputs)

        if batch_by_size:
            # create mini-batches with given size constraints
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )
        else:
            assert (
                max_sentences is not None
            ), "If batch_by_size=False, max_sentences must be passed. Got None"
            starts = indices[::max_sentences]
            batch_sampler = [indices[s:s + max_sentences] for s in starts]
        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter


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

        # TODO: fix hard code
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
                retvals[g][beam] = {
                    "tokens":
                        generated_ids[g, 1:],
                    "positional_scores":
                        torch.zeros_like(generated_ids[g, 1:],
                                         dtype=torch.float16)
                }
        return retvals
