import asyncio
import argparse
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
import json
import time
from typing import Any
import uuid

import alpa
from alpa.serve import run_controller, CONTROLLER_NAME
import ray
import torch

from llm_serving.generator import Generator
from llm_serving.service.constants import (
    NUM_BEAMS, NUM_RETURN_SEQ, ALPA_SERVE_PORT, USE_RECAPTCHA, KEYS_FILENAME)
from llm_serving.service.recaptcha import load_recaptcha
from llm_serving.service.utils import build_logger


@dataclass(order=True)
class PrioritizedItem:
    priority: int
    item: Any=field(compare=False)


GenerateItem = namedtuple("GenerateItem", ["uid", "return_queue", "data"])
LogprobsItem = namedtuple("LogprobsItem", ["uid", "return_queue", "data"])


class LangaugeModelWorker:
    def __init__(self,
                 model_name: str,
                 path: str,
                 torch_device: str,
                 tokenizer_name: str,
                 num_beams: int,
                 num_return_sequences: int,
                 use_recaptcha: bool,
                 max_seq_len: int = 1024,
                 max_batch_size: int = 4,
                 logprobs_past_cache_size_limit: int = 4,
                 batch_timeout: float = 1.0):

        self.logger = build_logger()
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_seq_len = max_seq_len

        # Batch queues
        self.max_bs = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_queue = asyncio.PriorityQueue()
        self.logprobs_past_cache = defaultdict(lambda: (0, None))
        self.logprobs_past_cache_size_limit = logprobs_past_cache_size_limit
        asyncio.get_event_loop().create_task(self.batch_loop())

        # Load model
        if num_beams > 1: # beam search is on, disable sampling
            do_sample = False
        else:
            do_sample = True

        self.generator = Generator(model_name,
                                   path,
                                   torch_device=torch_device,
                                   tokenizer_name=tokenizer_name,
                                   num_beams=num_beams,
                                   num_return_sequences=num_return_sequences,
                                   max_seq_len=self.max_seq_len,
                                   max_batch_size=self.max_bs,
                                   do_sample=do_sample)

        # Authentication
        self.allowed_api_keys = []
        self.recaptcha = load_recaptcha(use_recaptcha)
        if use_recaptcha:
            keys = json.load(open(KEYS_FILENAME, "r"))
            self.allowed_api_keys = keys["allowed_api_keys"]

    async def batch_loop(self):
        while True:
            pri_item = await self.request_queue.get()

            # Get the next batch
            generate_batch = []
            logprobs_item = None
            non_batch = []
            if isinstance(pri_item.item, GenerateItem):
                # Wait for batch opportunity
                await asyncio.sleep(self.batch_timeout)
                generate_batch.append(pri_item.item)

                while (not self.request_queue.empty() and
                       len(generate_batch) < self.max_bs):
                    pri_item = self.request_queue.get_nowait()
                    if isinstance(pri_item.item, GenerateItem):
                        generate_batch.append(pri_item.item)
                    else:
                        non_batch.append(pri_item)
                        break

                # Put non-batch items back to request queue
                for x in non_batch:
                    self.request_queue.put_nowait(x)
            elif isinstance(pri_item.item, LogprobsItem):
                logprobs_item = pri_item.item
            else:
                raise RuntimeError(f"Invalid item: {pri_item.item}")

            # Process this batch
            if generate_batch:
                args = {
                    "inputs": [],
                    "min_tokens": [],
                    "max_tokens": [],
                }
                for item in generate_batch:
                    args["inputs"].append(item.data["input"])
                    args["min_tokens"].append(item.data["min_tokens"])
                    args["max_tokens"].append(item.data["max_tokens"])
                    # FIXME: Now we assume all items have the same remaining args
                    for key in [
                        "temperature", "top_p", "n", "best_of", "echo",
                    ]:
                        args[key] = item.data[key]
                results = self.generator.generate(**args)
                for item, res in zip(generate_batch, results):
                    item.return_queue.put_nowait((item.uid, res))

            if logprobs_item:
                logprobs_past_cache = self.logprobs_past_cache
                arg = logprobs_item.data
                inputs = arg["input"]
                num_inputs = len(inputs)
                cache_id = arg["cache_id"]
                # do the actual generations
                output = self.generator.forward(inputs, cache_id, pasts=logprobs_past_cache)
                # add to or update the cache with newly computed values
                logprobs_past_cache[cache_id] = (time.time(), output.past_key_values)
                # delete oldest key in cache if cache too big
                if len(logprobs_past_cache) > self.logprobs_past_cache_size_limit:
                    oldest_key = min(list(logprobs_past_cache.keys()), key=lambda k: logprobs_past_cache[k][0])
                    del logprobs_past_cache[oldest_key]

                logits = output.logits[:num_inputs, -1]
                logprobs = torch.log_softmax(logits, dim=-1)
                top_logprobs, top_indices = logprobs.topk(arg["top_k"], dim=1)

                # return at most top_k tokens, e.g. if network limited
                return_dict = {
                    'logprobs': top_logprobs.cpu().tolist(),
                    'indices': top_indices.cpu().tolist()
                }
                # broadcast them back
                logprobs_item.return_queue.put_nowait((logprobs_item.uid, return_dict))

    async def handle_request(self, request):
        args = await request.json()
        self.check_authorization(args, request)

        if "completions" in request.url.path:
            return await self.completions(args, request)
        elif "logprobs" in request.url.path:
            return await self.logprobs(args, request)
        else:
            raise ValueError("Invalid url: {request.url}")

    def normalize_prompts(self, prompts):
        # prompt can be 4 types:
        # - case 1: str. Basic case. Return one generation.
        # - case 2: List[str]. Multiple generations, one per prompt.
        # - case 3: List[int]. Pretokenized. Return one generation.
        # - case 4: List[List[int]]. Pretokenized multiple generations.
        # our approach is to turn everything into the case 4
        if isinstance(prompts, str):  # case 1
            prompts = [self.generator.encode(prompts)]
        elif isinstance(prompts, list) and isinstance(prompts[0], str):  # case 2
            prompts = [self.generator.encode(p) for p in prompts]
        elif isinstance(prompts, list) and isinstance(prompts[0], int):  # case 3
            prompts = [prompts]
        else:  # case 4
            assert isinstance(prompts[0], list)
            assert isinstance(prompts[0][0], int)
        if len(prompts[0]) <= 0:
            raise ValueError("The prompt must be nonempty.")
        return prompts

    async def completions(self, args, request):
        logger = self.logger

        if "redirect_logprobs" in args:
            # A redirection to workaround some security settings.
            return await self.logprobs(args, request)

        # Normalize prompts
        prompts = args["prompt"]
        prompts = self.normalize_prompts(prompts)

        # Generation arguments
        args["min_tokens"] = int(args.get("min_tokens", 0))
        args["max_tokens"] = int(args.get("max_tokens", self.max_seq_len))

        if self.num_beams > 1:
            # if beam search is enabled, disable all sampling
            args["temperature"] = 0.0
            args["top_p"] = 0.0
        else:
            args["temperature"] = round(float(args.get("temperature", 1.0)), 1)
            args["top_p"] = round(float(args.get("top_p", 1.0)), 1)

        assert 0 <= args["top_p"] <= 1
        assert 0 <= args["temperature"]

        args["n"] = int(args.get("n", self.num_return_sequences))
        args["echo"] = bool(args.get("echo", False))
        args["best_of"] = self.num_beams

        if "stop" in args:
            raise NotImplementedError("The stop argument is not implemented")

        logger.info(f"Received new generate request: "
                    f"prompt length {[len(p) for p in prompts]}, "
                    f"max_len: {args.get('max_tokens', 0)}, "
                    f"temperature: {args['temperature']}, "
                    f"top_p: {args['top_p']}, "
                    f"api_key: {args.get('api_key', None)}, "
                    f"ip: {self.get_remote_ip(request)}, "
                    f"tstamp: {request.scope['tstamp']}")

        cur_len = max(len(p) for p in prompts)
        self.check_max_length_limit(cur_len, self.max_seq_len)

        # Push the requests to the batch queue
        return_queue = asyncio.Queue()
        for i, prompt in enumerate(prompts):
            data = {"input": prompt, **args}
            priority = 0
            self.request_queue.put_nowait(PrioritizedItem(
                priority, GenerateItem(i, return_queue, data)))

        unordered_results = []
        for i in range(len(prompts)):
            unordered_results.append(await return_queue.get())

        # Sort results by the original ordering
        reordered = sorted(unordered_results, key=lambda x: x[0])
        results = []
        for _, generations in reordered:
            results += generations

        # Transform the results into the openai format
        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "choices": [
                {
                    "text": result["text"],
                    # TODO: align with what OpenAI returns
                } for result in results
            ],
        }

    async def logprobs(self, args, request):
        logger = self.logger

        # Normalize prompts
        prompts = args["prompt"]
        prompts = self.normalize_prompts(prompts)

        # we're going to cache the keys for all the prompts in the request all together, so limit batch size
        assert len(prompts) <= self.max_bs, "Please submit a smaller batch"
        prompt_length = len(prompts[0])
        for prompt in prompts:
            assert len(prompt) == prompt_length, "All prompts must be the same length to work with current caching implementation"

        # Generation arguments
        args["min_tokens"] = int(args.get("min_tokens", 0))
        args["max_tokens"] = int(args.get("max_tokens", self.max_seq_len))

        args["top_k"] = int(args.get("top_k", 100000))

        args['top_p'] = -1
        args["temperature"] = -1
        args["n"] = int(args.get("n", self.num_return_sequences))

        logger.info(f"Received new logprobs request: "
                    f"prompt length {[len(p) for p in prompts]}, "
                    f"top_k: {args['top_k']}, "
                    f"api_key: {args.get('api_key', None)}, "
                    f"ip: {self.get_remote_ip(request)}, "
                    f"tstamp: {request.scope['tstamp']}")

        cur_len = max(len(p) for p in prompts)
        self.check_max_length_limit(cur_len, self.max_seq_len)

        # Push the request to the batch queue
        cache_id = args["cache_id"] if "cache_id" in args else str(uuid.uuid4())
        ret_queue = asyncio.Queue()
        data = {"input": prompts, "cache_id": cache_id, **args}
        self.request_queue.put_nowait(PrioritizedItem(
            50, LogprobsItem(0, ret_queue, data)))
        results = await ret_queue.get()
        return {
            "cache_id": cache_id,
            "logprobs": results[1]['logprobs'],
            "indices": results[1]['indices']
        }

    def check_max_length_limit(self, cur_len, max_len):
        if cur_len > max_len:
            self.logger.info(f"Rejected a request with max prompt length = {cur_len}.")
            raise ValueError(f"Your prompt length  = {cur_len} is too long. "
                             f"Please make sure len(prompt) + response length <= {max_len}. "
                             f"Since this is a public service, we have limited the max length supported. "
                             f"If you want to try longer sequence length, "
                             f"please consider hosting your own service using Alpa.")

    def check_authorization(self, args, request):
        if args.get("api_key", None) in self.allowed_api_keys:
            return

        if not self.recaptcha.verify(
                args.get("g-recaptcha-response", ""),
                request.client.host):
            self.logger.error(f"Rejected a request with invalid captcha.")
            raise ValueError("Invalid captcha. If you are using the website, please click the "
                             "\"I'm not a robot\" button. If you are using client APIs, please "
                             "contact alpa developers to get an API key.")

    def get_remote_ip(self, request):
        for x in request.scope['headers']:
            if x[0] == b"x-forwarded-for":
                v = x[1].decode()
                return v[:v.index(':')]
        return request.client.host


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--tokenizer", type=str)
    parser.add_argument("--no-recaptcha", action="store_true")
    parser.add_argument("--register-name", type=str, default="default")
    parser.add_argument("--ssl-keyfile", type=str)
    parser.add_argument("--ssl-certfile", type=str)
    args = parser.parse_args()

    ray.init(address="auto", namespace="alpa_serve")

    try:
        controller = ray.get_actor(CONTROLLER_NAME)
    except ValueError:
        controller = run_controller(args.host, ALPA_SERVE_PORT, "/",
                                    args.ssl_keyfile, args.ssl_certfile)

    group_id = 0
    controller.launch_mesh_group_manager.remote(group_id)
    t = controller.register_model.remote(
        args.register_name, LangaugeModelWorker,
        (args.model, args.path, args.torch_device, args.tokenizer, NUM_BEAMS, NUM_RETURN_SEQ,
         False if args.no_recaptcha else USE_RECAPTCHA),
        override=True)
    ray.get(t)
    t = controller.create_replica.remote(args.register_name, group_id)
    ray.get(t)

    while True:
        pass
