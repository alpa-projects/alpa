import argparse
from collections import defaultdict
import logging.handlers
import logging
import os
import queue
import random
import time
import threading
import traceback
import uuid

import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

from opt_serving.generator import Generator
from opt_serving.service.queue import PriorityQueueRingShard
from opt_serving.service.responses import OAIResponse
from opt_serving.service.utils import build_logger
from opt_serving.service.workers import WorkItem
from opt_serving.service.constants import (
    MAX_SEQ_LEN, MAX_BATCH_TOKENS, BATCHING_TIMEOUT_MS,
    MAX_BS, NUM_BEAMS, NUM_RETURN_SEQ, LOGPROBS_BATCHING_TIMEOUT_MS,
    LOGPROBS_PAST_CACHE_SIZE_LIMIT, LOGPROBS_PAST_CACHE_TIMEOUT)

app = Flask(__name__, template_folder='service')

# The global text generator
generator: Generator= None
# The request queues
generate_batch_queue = PriorityQueueRingShard()
logprobs_batch_queue = PriorityQueueRingShard()
# Past key caches
logprobs_past_cache = defaultdict(lambda: (0, None))
# Logging
logger = build_logger()

# Generation related global parameters
# These arguments affect the website html/ccs, so we set them as global vars
sampling_css = ""
num_beams = NUM_BEAMS
num_return_sequences = NUM_RETURN_SEQ


def batching_loop():
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via generate_batch_queue)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    generate_batch = []
    logprobs_batch = []

    generate_batch_timeout = 0
    while True:
        generate_batch, receive_item = generate_loop(
                generate_batch, timeout=generate_batch_timeout,
                max_bs=MAX_BS)

        if receive_item:
            generate_batch_timeout = BATCHING_TIMEOUT_MS
        else:
            generate_batch_timeout = 0

        # only process logprobs requests if we're not actively serving generate requests
        if not receive_item:
            logprobs_batch = logprobs_loop(
                    logprobs_batch, timeout=0, max_bs=MAX_BS)


def generate_loop(generate_batch, timeout, max_bs):
    receive_item = False
    try: # generate endpoint
        # for now, we only have 1 worker, so can always index to shard 0
        target_queue = generate_batch_queue.queue_shards[0].get_largest_queue()
        if target_queue:
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            receive_item = True
            generate_batch.append(item)
            logger.debug(f"Get item: {item} into batch")

            bs = len(generate_batch)
            if bs >= max_bs:
                raise queue.Empty
    except queue.Empty:
        logger.debug(f"Prepare to process generate batch: {generate_batch}")
        if generate_batch:
            request_args = {
                "inputs": [],
                "min_tokens": [],
                "max_tokens": [],
            }
            for work_item in generate_batch:
                ro = work_item.data
                request_args["inputs"].append(ro["input"])
                request_args["min_tokens"].append(ro["min_tokens"])
                request_args["max_tokens"].append(ro["max_tokens"])
                # assumption: everyone has the same remaining args
                for key in [
                    "temperature", "top_p", "n", "best_of", "echo",
                ]:
                    request_args[key] = ro[key]
            # do the actual generations
            generations = generator.generate(**request_args)
            # broadcast them back
            for work_item, gen in zip(generate_batch, generations):
                work_item.return_queue.put((work_item.uid, gen))

            generate_batch.clear()
    return generate_batch, receive_item


def logprobs_loop(logprobs_batch, timeout, max_bs):
    try: # logprobs endpoint
        target_queue = logprobs_batch_queue.queue_shards[0].get_largest_queue()
        if target_queue:
            item = target_queue.get(timeout=timeout / 1000)
            logprobs_batch.append(item)
            logger.debug(f"Get item: {item} into batch")

            bs = len(logprobs_batch)
            if bs >= max_bs:
                raise queue.Empty
    except:
        logger.debug(f"Prepare to process logprobs batch: {logprobs_batch}")
        if logprobs_batch:
            for work_item in logprobs_batch: # for loop over work items to keep it as batch size 1 for now, to avoid annoying logic with mixing past key values
                inputs = []
                cache_ids = []
                ro = work_item.data
                inputs.append(ro["input"])
                cache_ids.append(ro["cache_id"])
                # do the actual generations
                outputs = generator.forward(inputs, cache_ids, pasts=logprobs_past_cache)
                generations = []
                for cache_id, output in zip(cache_ids, outputs):
                    # add to or update the cache with newly computed values
                    logprobs_past_cache[cache_id] = (time.time(), output.past_key_values)
                    # delete oldest key in cache if cache too big
                    if len(logprobs_past_cache) > LOGPROBS_PAST_CACHE_SIZE_LIMIT:
                        oldest_key = min(list(logprobs_past_cache.keys()), key=lambda k: logprobs_past_cache[k][0])
                        del logprobs_past_cache[oldest_key]

                    last_position = min(output.logits.shape[1], len(inputs[0])) - 1
                    logprobs = torch.log_softmax(output.logits[0, last_position], dim=-1)
                    sorted_logprobs, sorted_indices = torch.sort(logprobs, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logprobs, dim=-1), dim=-1)
                    num_to_return = 1 + (cumulative_probs < ro['top_p']).sum().item()
                    # return at most top_k tokens, e.g. if network limited
                    num_to_return = min(num_to_return, ro["top_k"])
                    generations.append({
                        'logprobs': sorted_logprobs[:num_to_return].tolist(),
                        'indices': sorted_indices[:num_to_return].tolist()
                    })
                # broadcast them back
                for work_item, gen in zip(logprobs_batch, generations):
                    work_item.return_queue.put((work_item.uid, gen))

                logprobs_batch.clear()

    # clear old keys in logprobs_past_cache which haven't been used recently
    existing_cache_ids = list(logprobs_past_cache.keys())
    for cache_id in existing_cache_ids:
        if time.time() - logprobs_past_cache[cache_id][0] > LOGPROBS_PAST_CACHE_TIMEOUT:
            del logprobs_past_cache[cache_id]
            logger.debug(f"Clear past cache: {cache_id}")

    return logprobs_batch


def worker_main(model_name, path, torch_device):
    global generator
    global num_beams
    global sampling_css

    # Disable multithreading in tokenizers and torch, as different Flask threads
    # may then fight for resources.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)

    # Arguments check
    assert (
        (NUM_BEAMS >= 1 and NUM_RETURN_SEQ == 1) or
        (NUM_BEAMS == 1 and NUM_RETURN_SEQ >= 1)
    ), "either beam search or sampling can be enabled, not both (currently)"

    if num_beams > 1: # beam search is on, disable sampling
        sampling_css = 'display:none'
        do_sample = False
    else:
        do_sample = True

    generator = Generator(model_name,
                          path,
                          torch_device=torch_device,
                          num_beams=num_beams,
                          num_return_sequences=num_return_sequences,
                          do_sample=do_sample)
    batching_loop()


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # now you're handling non-HTTP exceptions only
    response = jsonify({
        "error": {
            "message": str(e),
            "type": "error",
            "stacktrace": traceback.format_tb(e.__traceback__),
        }
    })
    if isinstance(e, ValueError):
        response.status = 400
    else:
        response.status = 500
    return response


def normalize_prompts(prompts):
    # prompt can be 4 types:
    # - case 1: str. Basic case. Return one generation.
    # - case 2: List[str]. Multiple generations, one per prompt.
    # - case 3: List[int]. Pretokenized. Return one generation.
    # - case 4: List[List[int]]. Pretokenized multiple generations.
    # our approach is to turn everything into the case 4
    if isinstance(prompts, str):  # case 1
        prompts = [generator.encode(prompts)]
    elif isinstance(prompts, list) and isinstance(prompts[0], str):  # case 2
        prompts = [generator.encode(p) for p in prompts]
    elif isinstance(prompts, list) and isinstance(prompts[0], int):  # case 3
        prompts = [prompts]
    else:  # case 4
        assert isinstance(prompts[0], list)
        assert isinstance(prompts[0][0], int)
    if len(prompts[0]) <= 0:
        raise Exception("The prompt must be nonempty.")
    return prompts


@app.route("/completions", methods=["POST"])
def completions():
    check_model_loading()

    # Normalize prompts
    prompts = request.json["prompt"]
    prompts = normalize_prompts(prompts)

    # Generation arguments
    generation_args = request.json
    generation_args["min_tokens"] = int(
        generation_args.get("min_tokens", 0))
    generation_args["max_tokens"] = int(
        generation_args.get("max_tokens", MAX_SEQ_LEN))

    if num_beams > 1:
        # if beam search is enabled, disable all sampling
        generation_args["temperature"] = 0.0
        generation_args["top_p"] = 0.0
    else:
        generation_args["temperature"] = round(
            float(generation_args.get("temperature", 1.0)), 1)
        generation_args["top_p"] = round(
            float(generation_args.get("top_p", 1.0)), 1)

    assert 0 <= generation_args["top_p"] <= 1
    assert 0 <= generation_args["temperature"]

    generation_args["n"] = int(generation_args.get("n", num_return_sequences))
    generation_args["echo"] = bool(generation_args.get("echo", False))
    generation_args["best_of"] = num_beams

    if "stop" in generation_args:
        raise NotImplementedError("The stop argument is not implemented")

    logger.info(f"Received new generate request: "
	        f"prompt length {[len(p) for p in prompts]}, "
                f"max_len: {generation_args.get('max_tokens', 0)}, "
                f"temperature: {generation_args['temperature']}, "
                f"top_p: {generation_args['top_p']}, "
                f"ip: {request.remote_addr}")

    cur_len = max(len(p) for p in prompts)
    check_max_length_limit(cur_len, MAX_SEQ_LEN)

    # Push the request to the batch queue
    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args["max_tokens"]
        generate_batch_queue.put(
            WorkItem(len(prompt) + max_len, i, ret_queue, request_object))
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    # weirdly, openai returns to you a flat list if you gave multiple prompts
    reordered = sorted(unordered_results, key=lambda x: x[0])
    results = []
    for prompt, (_, generations) in zip(prompts, reordered):
        results += generations
    # transform the result into the openai format
    return OAIResponse(results).__dict__()


@app.route("/logprobs", methods=["POST"])
def logprobs():
    check_model_loading()

    # Normalize prompts
    prompts = request.json["prompt"]
    del request.json["prompt"]
    prompts = normalize_prompts(prompts)

    # Generation arguments
    generation_args = request.json
    generation_args["min_tokens"] = int(
        generation_args.get("min_tokens", 0))
    generation_args["max_tokens"] = int(
        generation_args.get("max_tokens", MAX_SEQ_LEN))

    generation_args["top_p"] = round(
        float(generation_args.get("top_p", 1.0)), 1)
    generation_args["top_k"] = int(generation_args.get("top_k", 100000))

    generation_args["temperature"] = -1
    generation_args["n"] = int(generation_args.get("n", num_return_sequences))

    logger.info(f"Received new logprobs request: "
	        f"prompt length {[len(p) for p in prompts]}, "
                f"top_p: {generation_args['top_p']}, "
                f"top_k: {generation_args['top_k']}.")

    cur_len = max(len(p) for p in prompts)
    check_max_length_limit(cur_len, MAX_SEQ_LEN)

    # Push the request to the batch queue
    cache_ids = [request.json["cache_id"]] if "cache_id" in request.json else [str(uuid.uuid4()) for _ in range(len(prompts))]
    ret_queue = queue.Queue()
    for i, (prompt, cache_id) in enumerate(zip(prompts, cache_ids)):
        request_object = {"input": prompt, "cache_id": cache_id,
                          **generation_args}
        item = WorkItem(len(prompt), i, ret_queue, request_object)
        logprobs_batch_queue.put(item
            )
    unordered_results = []
    for _ in prompts:
        unordered_results.append(ret_queue.get())
    # resort results by the original ordering
    reordered = sorted(unordered_results, key=lambda x: x[0])
    return jsonify({"cache_id": cache_ids[0], "logprobs": reordered[0][1]['logprobs'], "indices": reordered[0][1]['indices']})


@app.route("/")
def index():
    return render_template('index.html',
                           sampling_css=sampling_css,
                           num_return_sequences=num_return_sequences)


def check_max_length_limit(cur_len, max_len):
    if cur_len > max_len:
        logger.info(f"Rejected a prompt with length = {cur_len}.")
        raise Exception(f"Your prompt length  = {cur_len} is too long. "
                        f"Please make sure len(prompt) + response length <= {max_len}. "
                        f"Since this is a public service, we have limited the max length supported. "
                        f"If you want to try longer sequence length, "
                        f"please consider hosting your own service using Alpa.")


def check_model_loading():
    if generator is None:
        raise RuntimeError(
            "The server just restarted after regular maintenance. "
            "It is loading the model now, which can take several minutes. "
            "Please come back later. ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--serve-logprobs", action="store_true", default=False)
    parser.add_argument("--torch-device", type=str, default="cpu")
    args = parser.parse_args()

    thread = threading.Thread(target=worker_main,
                              args=(args.model, args.path, args.torch_device), daemon=True)
    thread.start()

    app.run(host="0.0.0.0", port=args.port, threaded=True)
