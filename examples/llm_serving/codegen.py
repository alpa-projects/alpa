"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
import argparse

import numpy as np
from transformers import AutoTokenizer

from llm_serving.model.wrapper import get_model

def main(args):
    # Load the tokenizer.
    if "codegen" in args.model:
        name = args.model.replace("alpa", "Salesforce")\
                         .replace("jax", "Salesforce")
        tokenizer = AutoTokenizer.from_pretrained(name, padding_side = "left")
        tokenizer.pad_token = 50256
    generate_params = {
        "do_sample": args.do_sample,
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences
    }
    
    # Load the model
    model = get_model(model_name=args.model,
                      path="/data/chris/opt_weights",
                      batch_size=args.n_prompts,
                      **generate_params)

    # Generate
    prompts = [
        "def helloworld():",
        "Create a function that solves the two sum problem",
        "Create a function that solves the fibonacci sequence problem",
        "Create an array that contains the numbers 1 to 10"
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
    parser.add_argument("--model", type=str, default="alpa/opt-1.3b")
    parser.add_argument('--do-sample', action='store_true')
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--num-return-sequences', type=int, default=1)
    parser.add_argument('--n-prompts', type=int, default=4)
    args = parser.parse_args()

    main(args)
