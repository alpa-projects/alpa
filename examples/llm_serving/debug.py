"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

from llm_serving.model.wrapper import get_model

def main(args):
    # Load the tokenizer.
    if "opt" in args.model:
        # We have to use the 30B version because other versions have some issues.
        # The 30B version works for all OPT models.
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b")
        tokenizer.add_bos_token = False
    elif "bloom" in args.model:
        name = args.model.replace("alpa", "bigscience")\
                         .replace("jax", "bigscience")
        tokenizer = AutoTokenizer.from_pretrained(name)

    generate_params = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences
    }
    
    # Load the model
    model = get_model(model_name=args.model,
                      path="~/opt_weights",
                      batch_size=args.n_prompts,
                      **generate_params)

    input_id_list = [
        # First batch
        [45942, 2866, 16, 5, 892, 9, 44042, 8],
        [100, 261, 23888, 2426, 16, 10, 21624, 12, 4310, 3034, 9744, 25526, 11],
        [133, 589, 9, 886, 6, 10817, 16, 10, 285],
        [5625, 16, 10, 205, 183, 8, 38, 236, 7],
        [2264, 16, 5, 7440, 9, 16673, 873, 24214, 116],
        # Second batch
        [32826, 16, 5, 812, 343, 9],
        [2264, 109, 47, 206, 59, 5, 499, 9, 28850, 1975, 37079, 116],
        [2264, 109, 47, 206, 59, 5, 3099, 9, 301, 116],
        [19195, 140, 16, 5, 394, 9],
        [534, 10311, 12, 246, 16, 10, 739, 2777, 1421, 14, 16, 4453, 9],
    ]

    for input_ids in input_id_list:
        l = len(input_ids)
        input_ids = torch.Tensor([input_ids]).long()
        output_ids = model.generate(input_ids=input_ids,
                                    max_length=15,
                                    **generate_params)
        print("Output ids:", output_ids[0][l:])
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        print("Outputs:\n" + 100 * '-')
        for i, output in enumerate(outputs):
            print(f"{i}: {output}")
            print(100 * '-')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpa/opt-1.3b")
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--n-prompts', type=int, default=1)
    args = parser.parse_args()

    main(args)
