if __name__ == "__main__":
    from test_text_gen import get_model, GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, time
    tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

    model = get_model("alpa/opt-175B", "cuda", False)
    prompt = "Computer science is the study of computation and"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    output = model.generate(input_ids=input_ids, max_length=20, do_sample=True)
    print(tokenizer.batch_decode(output, skip_special_tokens=True))
