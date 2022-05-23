import time
import logging

from typing import List, Optional, Dict
import torch
from torch import Tensor

from metaseq import checkpoint_utils, tasks

from examples.opt.serving import utils
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel
from transformers.generation_utils import GenerationMixin, ModelOutput, dataclass

logger = logging.getLogger(__name__)


class GeneratorInterface:
    """Alpa generator interface"""
    def __init__(self, cfg):
        self.cfg = cfg

        self.model = None
        self.inference_func = None
        self.task = None
        self.tokenizer = None

    def load_model(self, model_name="facebook/opt-125m"):
        """Load model and return the inference_func."""

        # Initialize model
        if "gpt" in model_name:
            self.model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda:0").half()
        elif "opt" in model_name:
            model_path = "/home/ubuntu/parax-efs/pycharm/opt/raw_weights/125M/pytorch_model.bin"
            self.model = OPTForCausalLM.from_pretrained(model_path).to("cuda:0").half()
        else:
            raise RuntimeError("Unrecognized model name.")

        def inference_func(input_ids, past_key_values):
            assert input_ids.shape[1] == 1, f"{input_ids.shape}"
            if past_key_values is None:
                attention_mask = None
            else:
                past_length = past_key_values[0][0].shape[2]
                attention_mask = torch.ones((input_ids.shape[0], past_length + 1)).to("cuda:0")
            out = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             past_key_values=past_key_values)
            return InferenceFuncOutput(out.logits, out.past_key_values)

        # init inference func
        self.inference_func = inference_func

        # Init tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Init task
        # TODO(Hao): remove this part later
        self.task = tasks.setup_task(self.cfg.task)

        return inference_func

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
        start_time = time.time()
        total_generation_time = 0

        # Initialize generator
        if not best_of:
            best_of = n
        assert best_of >= n
        self.cfg.generation.sampling_topp = top_p if top_p > 0 else -1
        self.cfg.generation.sampling = top_p > 0.0
        self.cfg.generation.beam = best_of
        if temperature > 0:
            self.cfg.generation.temperature = temperature
        else:
            self.cfg.generation.temperature = 1.0

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
            self.cfg.generation.min_len = total_min_tokens
            self.cfg.generation.max_len_b = total_max_tokens
            self.cfg.generation.max_len_a = 0

            logger.info(f"Preparing generator with settings {self.cfg.generation}")
            # generator = self.task.build_generator(
            #     self.models, self.cfg.generation, extra_gen_cls_kwargs={"stop": stop}
            # )
            generator = self.build_generator(self.inference_func, self.cfg.generation, extra_gen_cls_kwargs={"stop": stop})


            # okay actually generate
            logger.info(f"Executing generation on input tensor size {src_tokens.shape}")
            if use_cuda:
                batch = utils.move_to_cuda(batch)

            translate_start_time = time.time()
            translations = self.inference_step(generator, batch)
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
                    # text = self.bpe.bpe.decode(tokens)
                    # re-encode it so we get offsets
                    # token_offsets = [s for s, e in self.bpe.bpe.encode(text).offsets]
                    # token_offsets = [s for s, e in self.tokenizer.encode(text).offsets]

                    # result = {
                    #     "text": self.bpe.bpe.decode(tokens),
                    #     "tokens": [self.bpe.bpe.decode([t]) for t in tokens],
                    #     # text offset is useful for cutting off prompts or prefixes
                    #     # or evaluating PPL on just a subset of tokens
                    #     "text_offset": token_offsets,
                    #     "token_scores": scores,
                    #

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

    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        # Choose search strategy.
        sampling = getattr(args, "sampling", False)
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)
        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        if sampling:
            # search_strategy = search.Sampling(
            #     self.target_dictionary, sampling_topk, sampling_topp
            # )
            search_strategy = "sampling"
        else:
            search_strategy = "beam_search"

        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
        if seq_gen_cls is None:
            seq_gen_cls = WrappedInferenceFunc

        return seq_gen_cls(
            models,
            self.model.config,
            self.task.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            temperature=getattr(args, "temperature", 1.0),
            search_strategy=search_strategy,
            sampling=sampling,
            top_k=sampling_topk,
            top_p=sampling_topp,
            **extra_gen_cls_kwargs,
        )

    def inference_step(self, generator, sample, prefix_tokens=None, **kwargs):
        with torch.no_grad():
            # Generation will always be conditioned on bos_token
            if getattr(self.task.args, "add_bos_token", False):
                bos_token = self.task.source_dictionary.bos()
            else:
                bos_token = self.task.source_dictionary.eos()

            # SequenceGenerator doesn't use src_tokens directly, we need to
            # pass the `prefix_tokens` argument instead
            if prefix_tokens is None and sample["net_input"]["src_tokens"].nelement():
                prefix_tokens = sample["net_input"]["src_tokens"]
                if prefix_tokens[:, 0].eq(bos_token).all():
                    prefix_tokens = prefix_tokens[:, 1:]

            return generator.opt_generate(
                sample,
                prefix_tokens=prefix_tokens,
                bos_token=bos_token,
                **kwargs,
            )

@dataclass
class InferenceFuncOutput(ModelOutput):
    logits: any = None
    past_key_values: any = None


class WrappedInferenceFunc(GenerationMixin):
    """This class is a alternative of the SequenceGenerator class."""
    def __init__(self,
                 inference_func, # model
                 config, # required by hf
                 tgt_dict,
                 beam_size: int = 1,
                 max_len_a: int = 0,
                 max_len_b: int = 200,
                 min_len: int = 1,
                 temperature: float = 1.0,
                 search_strategy=None,
                 need_logprobs: bool = False,
                 stop: Optional[List[int]] = None,
                 sampling=False,
                 top_k=0.0,
                 top_p=0.0
                 ):
        self.inference_func = inference_func
        self.config = config
        self.main_input_name = "input_ids"

        # Params copied from SequenceGenerator
        self.tgt_dict = tgt_dict
        self.pad = tgt_dict.pad()
        self.unk = tgt_dict.unk()
        self.eos = tgt_dict.eos()
        self.vocab_size = len(tgt_dict)
        self.beam_size = beam_size
        # the max beam size is the dictionary size - 1, since we never select pad
        self.beam_size = min(beam_size, self.vocab_size - 1)
        self.max_len_a = max_len_a
        self.max_len_b = max_len_b
        self.min_len = min_len
        self.need_logprobs = need_logprobs
        self.stop = stop if stop is not None else []

        self.sampling = sampling
        self.top_k = top_k
        self.top_p = top_p

        self.temperature = temperature
        assert temperature > 0, "--temperature must be greater than 0"

        # self.search = (
        #     search.BeamSearch(tgt_dict) if search_strategy is None else search_strategy
        # )
        # self.model.eval()



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

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

    def opt_generate(self, sample: Dict[str, Dict[str, Tensor]], **kwargs):
        """Generate translations. Match the api of other metaseq generators."""
        return self._opt_generate(sample, **kwargs)

    def _opt_generate(
        self,
        sample: Dict[str, Dict[str, Tensor]],
        prefix_tokens: Optional[Tensor] = None,
        bos_token: Optional[int] = None,
    ):
        """
        Args:
            sample (dict): batch
            prefix_tokens (torch.LongTensor, optional): force decoder to begin
                with these tokens
            bos_token (int, optional): beginning of sentence token
                (default: self.eos)
        """
        # TODO(Hao): process sample into input_ids
        input_ids = sample["net_input"]["src_tokens"]

        generated_ids = self.generate(
            input_ids=input_ids,
            min_length=self.min_len,
            max_length=self.max_len_b,
            # temperature=self.temperature,
            # do_sample=self.sampling,
            # top_k=self.top_k,
            # top_p=self.top_p,
            # pad_token_id=self.pad,
            # eos_token_id=self.eos,
            # early_stopping=True,
            # no_repeat_ngram_size=2
            # return_dict_in_generate=True,
            # output_scores=True
            )
        retvals = [[{} for _ in range(self.beam_size)] for _ in generated_ids]
        for g in range(generated_ids.shape[0]):
            for beam in range(self.beam_size):
                retvals[g][beam] = {"tokens": generated_ids[g, 1:],
                                    "positional_scores": torch.zeros_like(generated_ids[g, 1:], dtype=torch.float16)}
        print(retvals)
        return retvals
