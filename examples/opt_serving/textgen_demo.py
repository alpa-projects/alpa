"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
from transformers import AutoTokenizer
from opt_serving.model.wrapper import get_model

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

generate_params = {'do_sample': True, 'num_beams': 4, 'num_return_sequences': 3}

# Load the model
model = get_model(model_name="alpa/opt-2.7b",
                  device="cuda",
                  path="/home/ubuntu/opt_weights",
                  **generate_params)

# Generate
prompt = "Paris is the capital city of"

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
outputs = model.generate(input_ids=input_ids,
                         max_length=256,
                         no_repeat_ngram_size=2,
                         **generate_params)

# Print results
print("Output:\n" + 100 * '-')
for i, output in enumerate(outputs):
    print("{}: {}".format(i, tokenizer.decode(output,
                                              skip_special_tokens=True)))
    print(100 * '-')
