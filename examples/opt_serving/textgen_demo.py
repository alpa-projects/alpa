"""Use huggingface/transformers interface and Alpa backend for distributed inference."""
from transformers import AutoTokenizer
from examples.opt_serving.model.wrapper import get_model

# Load the tokenizer. We have to use the 30B version because
# other versions have some issues. The 30B version works for all OPT models.
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
tokenizer.add_bos_token = False

# Load the model
model = get_model(model_name="alpa/opt-2.7b",
                  device="cuda",
                  path="/home/ubuntu/opt_weights/")

# Generate
prompt = "Paris is the capital city of "

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids=input_ids, max_length=256, do_sample=True)
generated_string = tokenizer.batch_decode(output, skip_special_tokens=True)

print(generated_string)
