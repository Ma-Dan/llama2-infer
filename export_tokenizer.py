# get all the tokens (postprocessed) and their scores as floats
import struct
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/Qwen1.5-0.5B-Chat", trust_remote_code=True)

tokens, scores = [], []
for i in range(151936):
    t = tokenizer.decode(i)
    s = 0
    b = t.encode('utf-8') # bytes of this token, utf-8 encoded
    tokens.append(b)
    scores.append(s)

max_token_length = max(len(t) for t in tokens)

print(max_token_length)

with open("qwen_chat_tokenizer.bin", 'wb') as f:
    f.write(struct.pack("I", max_token_length))
    for bytes, score in zip(tokens, scores):
        f.write(struct.pack("fI", score, len(bytes)))
        f.write(bytes)