"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
import argparse
import time

import numpy as np
from transformers import AutoTokenizer

from llm_serving.model.wrapper_1d import get_model
from alpa.timer import timers


def main(args):
    # Load the tokenizer. We have to use the 30B version because
    # other versions have some issues. The 30B version works for all OPT models.
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    tokenizer.add_bos_token = False

    generate_params = {
        "do_sample": args.do_sample,
        "max_length": 64
    }

    # Load the model
    model = get_model(model_name=args.model,
                      path="~/opt_weights")

    prompts = [
        "Computer science is the study of computation and",
        "Ion Stoica is a Romanian-American computer scientist specializing in",
        "The University of California, Berkeley is a public",
        "Today is a good day and I want to", "What is the valuation of Databricks?",
        "Paris is the capital city of", "Which country has the most population?",
        "What do you think about the future of Cryptocurrency?",
        "What do you think about the meaning of life?",
        "Donald Trump is the president of",
        "GPT-3 is a large language model that is capable of"
    ]

    # timer_names = ["enter", "compute", "update", "reshape"]

    input_ids = tokenizer(prompts, return_tensors="np", padding="longest").input_ids

    n_warmup = 10
    for i in range(n_warmup):
        tic = time.time()
        output_ids = model.generate(input_ids,
                                    **generate_params)
        elapsed = time.time() - tic
        print(f"- It takes {elapsed}")

        # Print results
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        if False:
            print("Outputs:\n" + 100 * '-')
            for i, output in enumerate(outputs):
                print(f"{i}: {output}")
                print(100 * '-')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-1d-125m")
    parser.add_argument('--do-sample', action='store_true')
    args = parser.parse_args()

    main(args)