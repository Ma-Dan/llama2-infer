from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen1.5-0.5B-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/Qwen1.5-0.5B-Chat", trust_remote_code = True)

batch = tokenizer("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n介绍一下自己<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt")

#batch = tokenizer("<|im_start|>", return_tensors="pt")

result = model.generate(batch["input_ids"].cpu(), do_sample=True, top_k=50, max_length=2048, top_p=0.95, temperature=0.8)
decoded = tokenizer.decode(result[0])
print(decoded)