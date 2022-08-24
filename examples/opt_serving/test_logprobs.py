import argparse
import time

import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer

from client import Client


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str)
    args = parser.parse_args()

    client = Client(args.url)
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    tokenizer.add_bos_token = False

    prompt = [int(x) for x in tokenizer.encode("Paris is the capital city of")]
    top_k = 50

    output = client.logprobs(prompt, top_k=top_k)

    tic = time.time()
    num_tokens = 40
    for i in range(num_tokens):
        distribution = np.full((tokenizer.vocab_size + 10), -1e8, dtype=np.float32)
        for idx, logprob in zip(output['indices'], output['logprobs']):
            distribution[idx] = logprob
        #distribution = softmax(distribution)
        #token = np.random.choice(np.arange(len(distribution)), p=distribution)
        token = distribution.argmax()
        prompt.append(int(token))
        print(tokenizer.decode(prompt))

        output = client.logprobs(prompt, top_k=top_k, cache_id=output["cache_id"])
    time_cost = time.time() - tic
    print(f"Generation throughput: {num_tokens/time_cost:.2f} token/s")

