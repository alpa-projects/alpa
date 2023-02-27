"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
import argparse

import numpy as np
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
                      path=args.path,
                      batch_size=args.n_prompts,
                      **generate_params)

    # Generate
    prompts = [
        "Paris is the capital city of",
        "Today is a good day and I'd like to",
        "Computer Science studies the area of",
        "University of California Berkeley is a public university"
    ]
    prompts = prompts[:args.n_prompts]
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
    parser.add_argument('--model', type=str, default='alpa/opt-1.3b')
    parser.add_argument('--path', type=str, default='~/opt_weights')
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--n-prompts', type=int, default=4)
    args = parser.parse_args()

    main(args)
