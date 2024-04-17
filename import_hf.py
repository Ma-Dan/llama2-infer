import torch
import os
from model import Transformer, ModelArgs

model_args = dict(
    dim=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=12,
    vocab_size=32000,
    hidden_dim=2268,
    max_seq_len=1024,
    dropout=0.0,
)

gptconf = ModelArgs(**model_args)

data = torch.load(f"/mnt/d/llama2/chinese-baby-llama2/pytorch_model.bin")

renamed_data = {}
model = {}
for k, v in data.items():
    k = k.replace("model.layers", "layers")
    k = k.replace(".mlp.gate_proj", ".feed_forward.w1")
    k = k.replace(".mlp.down_proj", ".feed_forward.w2")
    k = k.replace(".mlp.up_proj", ".feed_forward.w3")
    k = k.replace(".self_attn.q_proj", ".attention.wq")
    k = k.replace(".self_attn.k_proj", ".attention.wk")
    k = k.replace(".self_attn.v_proj", ".attention.wv")
    k = k.replace(".self_attn.o_proj", ".attention.wo")
    k = k.replace("model.norm.weight", "norm.weight")
    k = k.replace("lm_head.weight", "output.weight")
    k = k.replace("post_attention_layernorm.weight", "ffn_norm.weight")
    k = k.replace("input_layernorm.weight", "attention_norm.weight")
    k = k.replace("model.embed_tokens.weight", "tok_embeddings.weight")
    model[k] = v

renamed_data["model"] = model
renamed_data["model_args"] = model_args

torch.save(renamed_data, f"chinese-baby-llama2.pt")