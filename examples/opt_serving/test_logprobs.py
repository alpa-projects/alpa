import argparse

import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer

from opt_serving.client import Client


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

    for i in range(40):
        distribution = np.full((tokenizer.vocab_size + 10), -1e8, dtype=np.float32)
        for idx, logprob in zip(output['indices'], output['logprobs']):
            distribution[idx] = logprob
        #distribution = softmax(distribution)
        #token = np.random.choice(np.arange(len(distribution)), p=distribution)
        token = distribution.argmax()
        prompt.append(int(token))
        print(tokenizer.decode(prompt))

        output = client.logprobs(prompt, top_k=top_k, cache_id=output["cache_id"])
