import asyncio
import argparse
from collections import defaultdict, namedtuple
import logging
import time
import uuid

import alpa
from alpa.serve import CONTROLLER_NAME
import ray

from llm_serving.service.constants import NUM_BEAMS, NUM_RETURN_SEQ
from llm_serving.generator import Generator


GenerateItem = namedtuple("GenerateItem", ["uid", "return_queue", "data"])
LogprobsItem = namedtuple("LogprobsItem", ["uid", "return_queue", "data"])


class LangaugeModel:
    def __init__(self,
                 model_name: str,
                 path: str,
                 torch_device: str,
                 num_beams: int,
                 num_return_sequences: int,
                 batch_timeout: float = 1.0,
                 max_batch_size: int = 4,
                 max_seq_len: int = 1024):

        self.logger = logging.getLogger("LangaugeModel")
        self.num_beams = num_beams
        self.num_return_sequences = num_return_sequences
        self.max_seq_len = max_seq_len

        # Batch queues
        self.request_queue = asyncio.PriorityQueue()
        self.max_bs = max_batch_size
        self.batch_timeout = batch_timeout
        self.logprobs_past_cache = defaultdict(lambda: (0, None))
        asyncio.get_event_loop().create_task(self.batch_loop())

        # Load model
        self.generator = None
        asyncio.get_event_loop().create_task(self.load_model(
            model_name, path, torch_device, num_beams, num_return_sequences))

        # Authentication
        self.recaptcha = None
        self.allowed_api_keys = []

    async def load_model(self, model_name, path, torch_device,
                         num_beams, num_return_sequences):
        if num_beams > 1: # beam search is on, disable sampling
            do_sample = False
        else:
            do_sample = True

        self.generator = Generator(model_name,
                                   path,
                                   torch_device=torch_device,
                                   num_beams=num_beams,
                                   num_return_sequences=num_return_sequences,
                                   max_seq_len=self.max_seq_len,
                                   max_batch_size=self.max_bs,
                                   do_sample=do_sample)

    async def batch_loop(self):
        while True:
            _, item = await self.request_queue.get()

            # Get the next batch
            generate_batch = []
            logprobs_item = None
            non_batch = []
            if isinstance(item, GenerateItem):
                # Wait for batch opportunity
                await asyncio.sleep(self.batch_timeout / 1e3)
                generate_batch.append(item)

                while (not self.request_queue.empty() and
                       len(generate_batch) < self.max_bs):
                    item = self.request_queue.get_nowait()
                    if isinstance(item, GenerateItem):
                        generate_batch.append(item)
                    else:
                        non_batch.append(item)
                        break

                # Put non-batch items back to request queue
                for item in non_batch:
                    self.request_queue.put_nowait(item)
            elif isinstance(item, LogprobsItem):
                logprobs_item = item
            else:
                raise RuntimeError(f"Invalid item: {item}")

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
            elif logprobs_item:
                pass

    async def handle_request(self, request):
        args = await request.json()
        if "completions" in request.url.path:
            return await self.completions(args)
        elif "logprobs" in request.url.path:
            return await self.logprobs(args)
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
            raise Exception("The prompt must be nonempty.")
        return prompts

    async def completions(self, args):
        logger = self.logger

        if "redirect_logprobs" in args:
            # A redirection to workaround some security settings.
            return self.logprobs(args)

        self.check_model_loading()
        self.check_authorization(args)

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
                    f"api_key: {args.get('api_key', None)}, ")

        cur_len = max(len(p) for p in prompts)
        self.check_max_length_limit(cur_len, self.max_seq_len)

        # Push the requests to the batch queue
        return_queue = asyncio.Queue()
        for i, prompt in enumerate(prompts):
            data = {"input": prompt, **args}
            priority = 0
            self.request_queue.put_nowait(
                (priority, GenerateItem(i, return_queue, data)))

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

    async def logprobs(self, parameters):
        raise NotImplementedError

    def check_max_length_limit(self, cur_len, max_len):
        if cur_len > max_len:
            self.logger.info(f"Rejected a request with max prompt length = {cur_len}.")
            raise ValueError(f"Your prompt length  = {cur_len} is too long. "
                             f"Please make sure len(prompt) + response length <= {max_len}. "
                             f"Since this is a public service, we have limited the max length supported. "
                             f"If you want to try longer sequence length, "
                             f"please consider hosting your own service using Alpa.")

    def check_model_loading(self):
        if self.generator is None:
            self.logger.error(f"Rejected a request during model loading.")
            raise RuntimeError(
                "The server just restarted after regular maintenance. "
                "It is loading the model now, which can take several minutes. "
                "Please come back later. ")

    def check_authorization(self, args):
        if args.get("api_key", None) in self.allowed_api_keys:
            return

        if self.recaptcha and not self.recaptcha.verify():
            logger.error(f"Rejected a request with invalid captcha.")
            raise ValueError("Invalid captcha. If you are using the website, please click the "
                             "\"I'm not a robot\" button. If you are using client APIs, please "
                             "contact alpa developers to get an API key.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--torch-device", type=str, default="cpu")
    parser.add_argument("--use-recaptcha", action="store_true")
    parser.add_argument("--keys-file", type=str, default="keys.json")
    args = parser.parse_args()

    ray.init(address="auto", namespace="alpa_serve")

    controller = ray.get_actor(CONTROLLER_NAME)

    group_id = 0
    name = "default"
    controller.launch_mesh_group_manager.remote(group_id)
    t = controller.register_model.remote(
        name, LangaugeModel,
        (args.model, args.path, args.torch_device, NUM_BEAMS, NUM_RETURN_SEQ),
        override=True)
    a = controller.create_replica.remote(name, group_id)
    ray.get(a)
