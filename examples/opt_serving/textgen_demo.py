"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
from transformers import AutoTokenizer
from opt_serving.model.wrapper import get_model
import numpy as np

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

generate_params = {"do_sample": True, "num_beams": 1, "num_return_sequences": 1}

# Load the model
model = get_model(model_name="alpa/opt-125m",
                  device="cuda",
                  path="/home/ubuntu/opt_weights",
                  batch_size=4,
                  max_target_positions=512,
                  **generate_params)

# Generate
prompt = [
    "Paris is the capital city of",
    "Today is a good day and I'd like to",
    "Computer Science studies the area of",
    "University of California Berkeley is a public university"
]
input_ids = tokenizer(prompt, return_tensors="pt", padding="longest").input_ids.to("cuda")

outputs = model.generate(input_ids=input_ids,
                         pad_token_id=tokenizer.pad_token_id,
                         max_length=64,
                         **generate_params)

# Print results
print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(output,
                                              skip_special_tokens=True)))
    print(100 * '-')
