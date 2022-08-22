import logging

import argparse
import queue
import threading
import os
import logging.handlers
import traceback

import random
import torch
from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import HTTPException

from opt_serving.generator import GeneratorInterface
from opt_serving.service.queue import PriorityQueueRingShard
from opt_serving.service.responses import OAIResponse
from opt_serving.service.utils import encode_fn, build_logger
from opt_serving.service.workers import WorkItem
from opt_serving.service.constants import MAX_SEQ_LEN, MAX_BATCH_TOKENS, TIMEOUT_MS, \
    MAX_BS, NUM_BEAMS, NUM_RETURN_SEQ

app = Flask(__name__, template_folder='service')

# The global text generator
generator: GeneratorInterface = None
# The request queue
batch_queue = PriorityQueueRingShard()
# Logging
logger = build_logger()

# Generation related global parameters
# These arguments affect the website html/ccs, so we set them as global vars
sampling_css = ""
num_beams = NUM_BEAMS
num_return_sequences = NUM_RETURN_SEQ


def batching_loop(timeout=TIMEOUT_MS, max_tokens=MAX_BATCH_TOKENS, max_bs=MAX_BS):
    """
    batching_loop is an infinite loop responsible for executing generations.

    GPUs benefit from batching requests, but we expect workloads to come
    in non-uniformly. This loop groups requests together (via batch_queue)
    and executes them in one batch. In order to keep latency low, unfilled
    batches are executed within a window of :timeout: milliseconds.

    batching_loop also performs dynamic batching, in order to minimize the
    amount of padding by grouping like-sized workloads together. As a result
    batching loop will provide preferential treatment to smaller workloads.  At
    the current moment, there is no TTL logic to ensure a maximum wait time.

    For a rough overview of dynamic batching, see
    https://parl.ai/docs/tutorial_worlds.html#dynamic-batching.

    :param timeout: The max queue time before a non-full batch is launched.
    :param max_tokens: the maximum number of tokens that can be processed
        concurrently. model specific and empirical.
    """
    # TODO(roller):
    # - group by generation type, topp etc, as we cannot share these
    # - modify timeout logic to be cumulative
    batch = []
    while True:
        try:
            # for now, we only have 1 worker, so can always index to shard 0
            target_queue = batch_queue.queue_shards[0].get_largest_queue()
            if not target_queue:
                continue
            # dynamic batching: group like-sized items to reduce the cost
            # of padding. See PR#20 for additional context.
            item = target_queue.get(timeout=timeout / 1000)
            logger.debug(f"Get item: {item} into batch")
            # accumulate the batch until it gets too big
            # Below we use number of tokens as a measure to accumulate batch
            # longest = max([item] + batch).cost
            # batch_cost = longest * (len(batch) + 1)
            # Below we use number of sequences as a measure to accumulate batch
            bs = len(batch) + 1
            if batch and bs > max_bs:
                # we're over budget, put it back in the queue
                target_queue.put(item)
                raise queue.Empty
            else:
                # batch is empty or under budget
                batch.append(item)
        except queue.Empty:
            logger.debug(f"Prepare to process batch: {batch}")
            if batch:
                request_object = {
                    "inputs": [],
                    "min_tokens": [],
                    "max_tokens": [],
                }
                for work_item in batch:
                    ro = work_item.data
                    request_object["inputs"].append(ro["input"])
                    request_object["min_tokens"].append(ro.get("min_tokens", 0))
                    request_object["max_tokens"].append(
                        ro.get("max_tokens", MAX_SEQ_LEN))
                    # assumption: everyone has the same remaining args
                    for key in [
                            "temperature",
                            "top_p",
                            "n",
                            "best_of",
                            "echo",
                            "logprobs",
                            "stop",
                    ]:
                        if key in ro:
                            request_object[key] = ro[key]
                # do the actual generations
                generations = generator.generate(**request_object)
                # broadcast them back
                for work_item, gen in zip(batch, generations):
                    work_item.return_queue.put((work_item.uid, gen))

                batch.clear()
            else:
                # back to the loop
                continue


def worker_main(model_name, path, port, torch_device):
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

    generator = GeneratorInterface(model_name,
                                   path,
                                   torch_device=torch_device,
                                   num_beams=num_beams,
                                   num_return_sequences=num_return_sequences,
                                   do_sample=do_sample)

    thread = threading.Thread(target=batching_loop, daemon=True)
    thread.start()
    app.run(host="0.0.0.0", port=port, threaded=True)


@app.errorhandler(Exception)
def handle_exception(e):
    # pass through HTTP errors
    if isinstance(e, HTTPException):
        return e
    # now you're handling non-HTTP exceptions only
    response = jsonify({
        "error": {
            "message": str(e),
            "type": "oops",
            # "stacktrace": traceback.format_tb(e.__traceback__),
        }
    })
    if isinstance(e, ValueError):
        response.status = 400
    else:
        response.status = 500
    return response


@app.route("/completions", methods=["POST"])
def completions(engine=None):
    # prompt can be 4 types:
    # - str. Basic case. Return one generation.
    # - list of str. Multiple generations, one per prompt
    # - list of ints. Pretokenized. Return one generation
    # - list of list of ints. Pretokenized multiple generations.
    # our approach is to turn everything into the last case
    prompts = request.json["prompt"]
    del request.json["prompt"]
    generation_args = request.json
    if isinstance(prompts, str):
        # single string. tokenize and turn it to the single pre-tokenized case
        prompts = [encode_fn(generator, prompts)]
    assert isinstance(prompts, list)
    assert len(prompts) > 0
    if isinstance(prompts[0], str):
        # multi string
        prompts = [encode_fn(generator, p) for p in prompts]
    elif isinstance(prompts[0], int):
        # single pre-tokenized
        prompts = [prompts]
    assert isinstance(prompts[0], list)
    # final case: multi pre-tokenized
    if len(prompts[0]) <= 0:
        raise Exception("The prompt must be nonempty.")

    if "min_tokens" in generation_args:
        generation_args["min_tokens"] = int(generation_args["min_tokens"])
    if "max_tokens" in generation_args:
        generation_args["max_tokens"] = int(generation_args["max_tokens"])
    if "stop" in generation_args:
        stop = generation_args["stop"]
        if stop is None:
            pass
        elif isinstance(stop, str):
            stop = [encode_fn(generator, stop)[0]]
        else:
            stop = [encode_fn(generator, s)[0] for s in stop]
        generation_args["stop"] = stop
        raise NotImplementedError("The stop argument is not implemented")

    # beam search top n
    generation_args["best_of"] = num_beams
    if "n" in generation_args:
        generation_args["n"] = int(generation_args["n"])
    else:
        generation_args["n"] = num_return_sequences

    if num_beams > 1:
        # if beam search is enabled, disable all sampling
        generation_args["temperature"] = 0.0
        generation_args["top_p"] = 0.0
    else:
        if "temperature" in generation_args:
            generation_args["temperature"] = round(
                float(generation_args["temperature"]), 1)
        else:
            generation_args["temperature"] = 1.0
        if "top_p" in generation_args:
            generation_args["top_p"] = round(float(generation_args["top_p"]), 1)
        else:
            generation_args["top_p"] = 1.0

    logger.info(f"Received new request: prompt length {len(prompts[0])}, "
                f"max_len: {generation_args.get('max_tokens', 0)}, "
                f"temperature: {generation_args['temperature']}, "
                f"top_p: {generation_args['top_p']}.")
    if len(prompts[0]) + generation_args.get("max_tokens", 0) > MAX_SEQ_LEN:
        logger.info(f"Rejected a prompt with prompt length {len(prompts[0])}.")
        raise Exception("Your prompt length is too long. Please make sure len(prompt) + response length <= 512. "
                        "Since this is a public service, we have limited the max length supported. "
                        "If you want to try longer sequence length, please consider hosting your own service using Alpa.")

    # Push the request to the batch queue
    ret_queue = queue.Queue()
    for i, prompt in enumerate(prompts):
        request_object = {"input": prompt, **generation_args}
        max_len = generation_args.get("max_tokens", 0)
        batch_queue.put(
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


@app.route("/")
def index():
    return render_template('index.html',
                           sampling_css=sampling_css,
                           num_return_sequences=num_return_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-125m")
    parser.add_argument("--path", type=str, default="~/opt_weights/")
    parser.add_argument("--port", type=int, default=20001)
    parser.add_argument("--torch-device", type=str, default="cpu")
    args = parser.parse_args()
    worker_main(args.model, args.path, args.port, args.torch_device)
