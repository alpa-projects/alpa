"""
Usage:

python3 test_logprobs.py --url http://localhost:20001
python3 test_logprobs.py --url https://opt.alpa.ai --api-key YOUR_KEY
"""
import argparse
import time

import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer

from client import Client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    parser.add_argument("--api-key", type=str)
    args = parser.parse_args()

    client = Client(args.url, api_key=args.api_key)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    tokenizer.add_bos_token = False

    prompts = [
        "Paris is the capital city of France",
        "Computer science is the",
    ]

    input_ids = tokenizer(prompts, padding="longest").input_ids
    top_k = 50

    output = client.logprobs(input_ids, top_k=top_k)

    tic = time.time()
    num_tokens = 40
    for i in range(num_tokens):
        print("=" * 20 + f" Step {i} " + "=" * 20)
        for j in range(len(input_ids)):
            distribution = np.full((tokenizer.vocab_size + 10), -1e8, dtype=np.float32)
            for idx, logprob in zip(output['indices'][j], output['logprobs'][j]):
                distribution[idx] = logprob
            # distribution = softmax(distribution)
            # token = np.random.choice(np.arange(len(distribution)), p=distribution)
            token = distribution.argmax()
            input_ids[j].append(int(token))
            print(tokenizer.decode(input_ids[j], skip_special_tokens=True))
            print("-" * 20)
        output = client.logprobs(input_ids, top_k=top_k, cache_id=output["cache_id"])
    time_cost = time.time() - tic
    print(f"Generation throughput: {len(prompts) * num_tokens/time_cost:.2f} token/s")
