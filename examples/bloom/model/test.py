from transformers import AutoTokenizer
from wrapper import get_model


tokenizer = AutoTokenizer.from_pretrained("/home/x/Desktop/dedong_2022_summer/softwares/bloom-350m", use_fast=False)
# tokenizer.add_bos_token = False
# generate_params = {"do_sample": True, "num_beams": 1, "num_return_sequences": 1}
generate_params = {"do_sample": True}
prompt = "good day"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cpu")
print(input_ids.shape)
# config = BloomConfig
# model = flaxBloom.FlaxBloomForCausalLM.from_pretrained("/home/x/Desktop/dedong_2022_summer/softwares/bloom-350m", config=config)
model = get_model("jax/bloom-350m", "/home/x/Desktop/dedong_2022_summer/softwares/350M_Bloom_np", torch_device="cpu", **generate_params)
print("generate")
# logits = model(input_ids=input_ids)[1]
# with open("../../logits_FlaxBloom_cpu.txt", "w") as logits_out:
#     logits_out.write(logits + "\n")
output_ids = model.generate(input_ids=input_ids,
                        max_length=30,
                        **generate_params)
print(output_ids)
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
print(outputs)