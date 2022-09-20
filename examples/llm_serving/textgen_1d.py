"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
import argparse
import time

import numpy as np
from transformers import AutoTokenizer

from llm_serving.model.wrapper import get_model_1d


def main(args):
    # Load the tokenizer. We have to use the 30B version because
    # other versions have some issues. The 30B version works for all OPT models.
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
    tokenizer.add_bos_token = False

    generate_params = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences
    }

    # Load the model
    model = get_model_1d(model_name=args.model,
                         path="~/opt_weights")

    # Generate
    prompts = [
        "Paris is the capital city of",
        "Today is a good day and I'd like to",
        # "Computer Science studies the area of",
        # "University of California Berkeley is a public university",
    ]
    input_ids = tokenizer(prompts, return_tensors="pt", padding="longest").input_ids
    output_ids = model.generate(input_ids=input_ids,
                                max_length=64,
                                **generate_params)
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Print results
    print("Outputs:\n" + 100 * '-')
    for i, output in enumerate(outputs):
        print(f"{i}: {output}")
        print(100 * '-')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-1d-125m")
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    args = parser.parse_args()

    main(args)
