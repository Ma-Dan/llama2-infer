import struct
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

def input_node(param, node_name, out_name):
    param.append("Input_t {} 0 1 {}".format(node_name, out_name))

def embed_node(param, node_name, input_name, dim0, dim1, idx, weight_offset):
    param.append("Embed_t {} 1 1 {} {} 0={} 1={} 2={}".format(node_name, input_name, idx, dim0, dim1, weight_offset))
    return idx, weight_offset+dim0*dim1

def unary_node(param, node_name, operate, idx):
    idx_output = idx+1
    param.append("UnaryOp_t {} 1 1 {} {} 0={}".format(node_name, idx, idx_output, operate))
    return idx_output

def reduction_node(param, node_name, operate, idx):
    idx_output = idx+1
    param.append("Reduction_t {} 1 1 {} {} 0={} 1=1".format(node_name, idx, idx_output, operate))
    return idx_output

def binary1_node(param, node_name, operate, value, idx):
    idx_output = idx+1
    param.append("BinaryOp_t {} 1 1 {} {} 0={} 1=Param 2={}".format(node_name, idx, idx_output, operate, value))
    return idx_output

def binary2_node(param, node_name, operate, idx1, idx2):
    idx_output = max(idx1, idx2)+1
    param.append("BinaryOp_t {} 2 1 {} {} {} 0={} 1=Tensor".format(node_name, idx1, idx2, idx_output, operate))
    return idx_output

def memorydata1_node(param, node_name, dim0, weight_offset, idx):
    idx_output = idx+1
    param.append("MemoryData_t {} 0 1 {} 0=1 1={} 2={}".format(node_name, idx_output, dim0, weight_offset))
    return idx_output, weight_offset+dim0

def memorydata2_node(param, node_name, dim0, dim1, weight_offset, idx):
    idx_output = idx+1
    param.append("MemoryData_t {} 0 1 {} 0=2 1={} 2={} 3={}".format(node_name, idx_output, dim0, dim1, weight_offset))
    return idx_output, weight_offset+dim0*dim1

def matmul_node(param, node_name, idx1, idx2):
    idx_output = max(idx1, idx2)+1
    param.append("Matmul_t {} 2 1 {} {} {}".format(node_name, idx1, idx2, idx_output))
    return idx_output

def reshape3_node(param, node_name, dim0, dim1, dim2, idx):
    idx_output = idx+1
    param.append("Reshape_t {} 1 1 {} {} 0=3 1={} 2={} 3={}".format(node_name, idx, idx_output, dim0, dim1, dim2))
    return idx_output

def reshape2_node(param, node_name, dim0, dim1, idx):
    idx_output = idx+1
    param.append("Reshape_t {} 1 1 {} {} 0=2 1={} 2={}".format(node_name, idx, idx_output, dim0, dim1))
    return idx_output

def posenc_node(param, node_name, use_last, idx):
    idx_output = idx+1
    param.append("Posenc_t {} 3 1 {} freqs_cos freqs_sin {} 0={}".format(node_name, idx, idx_output, use_last))
    return idx_output

def concat_node(param, node_name, input_name, idx):
    idx_output = idx+1
    param.append("Concat_t {} 2 1 {} {} {} 0=0".format(node_name, input_name, idx, idx_output))
    return idx_output

def softmax_node(param, node_name, idx):
    idx_output = idx+1
    param.append("Softmax_t {} 1 1 {} {} 0=0".format(node_name, idx, idx_output))
    return idx_output

def swish_node(param, node_name, idx):
    idx_output = idx+1
    param.append("Swish_t {} 1 1 {} {}".format(node_name, idx, idx_output))
    return idx_output

def transformer_layer(model, param, layer_idx, idx_input, weight_offset):
    rms_norm_eps = model.config.rms_norm_eps
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    attention_dim = hidden_size // num_attention_heads
    intermediate_size = model.config.intermediate_size

    #attention norm
    idx_attnorm_square = unary_node(param, "layer{}_attnorm_square".format(layer_idx), "Square", idx_input)
    idx_attnorm_mean = reduction_node(param, "layer{}_attnorm_mean".format(layer_idx), "Mean", idx_attnorm_square)
    idx_attnorm_add = binary1_node(param, "layer{}_attnorm_add".format(layer_idx), "Add", rms_norm_eps, idx_attnorm_mean)
    idx_attnorm_rsq = unary_node(param, "layer{}_attnorm_rsq".format(layer_idx), "Rsq", idx_attnorm_add)
    idx_attnorm_mul1 = binary2_node(param, "layer{}_attnorm_mul1".format(layer_idx), "Mul", idx_input, idx_attnorm_rsq)
    idx_attnorm_weight, weight_offset = memorydata1_node(param, "layer{}_attnorm_weight".format(layer_idx), hidden_size, weight_offset, idx_attnorm_mul1)
    idx_attnorm_mul2 = binary2_node(param, "layer{}_attnorm_mul2".format(layer_idx), "Mul", idx_attnorm_mul1, idx_attnorm_weight)

    #Q
    idx_att_wq_weight, weight_offset = memorydata2_node(param, "layer{}_att_wq_weight".format(layer_idx), hidden_size, hidden_size, weight_offset, idx_attnorm_mul2)
    idx_att_q_linear = matmul_node(param, "layer{}_att_q_linear".format(layer_idx), idx_attnorm_mul2, idx_att_wq_weight)
    idx_att_q_reshape = reshape3_node(param, "layer{}_att_q_reshape".format(layer_idx), -1, num_attention_heads, attention_dim, idx_att_q_linear)
    idx_att_q_posenc = posenc_node(param, "layer{}_att_q_posenc".format(layer_idx), 1, idx_att_q_reshape)

    #K
    idx_att_wk_weight, weight_offset = memorydata2_node(param, "layer{}_att_wk_weight".format(layer_idx), hidden_size, hidden_size, weight_offset, idx_att_q_posenc)
    idx_att_k_linear = matmul_node(param, "layer{}_att_k_linear".format(layer_idx), idx_attnorm_mul2, idx_att_wk_weight)
    idx_att_k_concat = concat_node(param, "layer{}_att_k_concat".format(layer_idx), "k_cache_{}".format(layer_idx), idx_att_k_linear)
    idx_att_k_reshape = reshape3_node(param, "layer{}_att_k_reshape".format(layer_idx), -1, num_attention_heads, attention_dim, idx_att_k_concat)
    idx_att_k_posenc = posenc_node(param, "layer{}_att_k_posenc".format(layer_idx), 0, idx_att_k_reshape)

    #QK
    idx_att_qk_matmul = matmul_node(param, "layer{}_att_qk_matmul".format(layer_idx), idx_att_k_posenc, idx_att_q_posenc)
    idx_att_qk_div = binary1_node(param, "layer{}_att_qk_div".format(layer_idx), "Div", int(attention_dim**0.5), idx_att_qk_matmul)
    idx_att_qk_softmax = softmax_node(param, "layer{}_att_qk_softmax".format(layer_idx), idx_att_qk_div)

    #V
    idx_att_wv_weight, weight_offset = memorydata2_node(param, "layer{}_att_wv_weight".format(layer_idx), hidden_size, hidden_size, weight_offset, idx_att_qk_softmax)
    idx_att_v_linear = matmul_node(param, "layer{}_att_v_linear".format(layer_idx), idx_attnorm_mul2, idx_att_wv_weight)
    idx_att_v_concat = concat_node(param, "layer{}_att_v_concat".format(layer_idx), "v_cache_{}".format(layer_idx), idx_att_v_linear)
    idx_att_v_reshape = reshape3_node(param, "layer{}_att_v_reshape".format(layer_idx), -1, num_attention_heads, attention_dim, idx_att_v_concat)

    #QKV
    idx_att_qkv_matmul = matmul_node(param, "layer{}_att_qkv_matmul".format(layer_idx), idx_att_v_reshape, idx_att_qk_softmax)
    idx_att_qkv_reshape = reshape2_node(param, "layer{}_att_qkv_reshape".format(layer_idx), -1, hidden_size, idx_att_qkv_matmul)

    #WO
    idx_att_wo_weight, weight_offset = memorydata2_node(param, "layer{}_att_wo_weight".format(layer_idx), hidden_size, hidden_size, weight_offset, idx_att_qkv_reshape)
    idx_att_wo_matmul = matmul_node(param, "layer{}_att_wo_matmul".format(layer_idx), idx_att_qkv_reshape, idx_att_wo_weight)
    idx_att_shortcut = binary2_node(param, "layer{}_att_shortcut".format(layer_idx), "Add", idx_input, idx_att_wo_matmul)

    #FFN norm
    idx_ffnnorm_square = unary_node(param, "layer{}_ffnnorm_square".format(layer_idx), "Square", idx_att_shortcut)
    idx_ffnnorm_mean = reduction_node(param, "layer{}_ffnnorm_mean".format(layer_idx), "Mean", idx_ffnnorm_square)
    idx_ffnnorm_add = binary1_node(param, "layer{}_ffnnorm_add".format(layer_idx), "Add", rms_norm_eps, idx_ffnnorm_mean)
    idx_ffnnorm_rsq = unary_node(param, "layer{}_ffnnorm_rsq".format(layer_idx), "Rsq", idx_ffnnorm_add)
    idx_ffnnorm_mul1 = binary2_node(param, "layer{}_ffnnorm_mul1".format(layer_idx), "Mul", idx_att_shortcut, idx_ffnnorm_rsq)
    idx_ffnnorm_weight, weight_offset = memorydata1_node(param, "layer{}_ffnnorm_weight".format(layer_idx), hidden_size, weight_offset, idx_ffnnorm_mul1)
    idx_ffnnorm_mul2 = binary2_node(param, "layer{}_ffnnorm_mul2".format(layer_idx), "Mul", idx_ffnnorm_mul1, idx_ffnnorm_weight)

    #FFN
    idx_ffn_w1_weight, weight_offset = memorydata2_node(param, "layer{}_ffn_w1_weight".format(layer_idx), intermediate_size, hidden_size, weight_offset, idx_ffnnorm_mul2)
    idx_ffn_w1_matmul = matmul_node(param, "layer{}_ffn_w1_matmul".format(layer_idx), idx_ffnnorm_mul2, idx_ffn_w1_weight)
    idx_ffn_w1_swish = swish_node(param, "layer{}_ffn_w1_swish".format(layer_idx), idx_ffn_w1_matmul)
    idx_ffn_w3_weight, weight_offset = memorydata2_node(param, "layer{}_ffn_w3_weight".format(layer_idx), intermediate_size, hidden_size, weight_offset, idx_ffn_w1_swish)
    idx_ffn_w3_matmul = matmul_node(param, "layer{}_ffn_w3_matmul".format(layer_idx), idx_ffnnorm_mul2, idx_ffn_w3_weight)
    idx_ffn_mul = binary2_node(param, "layer{}_ffn_mul".format(layer_idx), "Mul", idx_ffn_w1_swish, idx_ffn_w3_matmul)
    idx_ffn_w2_weight, weight_offset = memorydata2_node(param, "layer{}_ffn_w2_weight".format(layer_idx), hidden_size, intermediate_size, weight_offset, idx_ffn_mul)
    idx_ffn_w2_matmul = matmul_node(param, "layer{}_ffn_w2_matmul".format(layer_idx), idx_ffn_mul, idx_ffn_w2_weight)
    idx_ffn_shortcut = binary2_node(param, "layer{}_ffn_shortcut".format(layer_idx), "Add", idx_att_shortcut, idx_ffn_w2_matmul)

    return idx_ffn_shortcut, weight_offset, idx_att_k_concat, idx_att_v_concat

def output_norm(model, param, idx_input, weight_offset):
    rms_norm_eps = model.config.rms_norm_eps
    hidden_size = model.config.hidden_size

    idx_outnorm_square = unary_node(param, "outnorm_square", "Square", idx_input)
    idx_outnorm_mean = reduction_node(param, "outnorm_mean", "Mean", idx_outnorm_square)
    idx_outnorm_add = binary1_node(param, "outnorm_add", "Add", rms_norm_eps, idx_outnorm_mean)
    idx_outnorm_rsq = unary_node(param, "outnorm_rsq", "Rsq", idx_outnorm_add)
    idx_outnorm_mul1 = binary2_node(param, "outnorm_mul1", "Mul", idx_input, idx_outnorm_rsq)
    idx_outnorm_weight, weight_offset = memorydata1_node(param, "outnorm_weight", hidden_size, weight_offset, idx_outnorm_mul1)
    idx_outnorm_mul2 = binary2_node(param, "attnorm_mul2", "Mul", idx_outnorm_mul1, idx_outnorm_weight)

    return idx_outnorm_mul2, weight_offset

def model_param(model, param):
    num_hidden_layers = model.config.num_hidden_layers
    vocab_size = model.config.vocab_size
    hidden_size = model.config.hidden_size

    #文件头
    param.append("7767517")
    param.append("8 1") #用来存模型基本参数和KVCache输出节点值

    #输入节点
    input_node(param, "input", "in")
    input_node(param, "freqs_cos", "freqs_cos")
    input_node(param, "freqs_sin", "freqs_sin")
    for i in range(num_hidden_layers):
        kcache_name = "k_cache_{}".format(i)
        input_node(param, kcache_name, kcache_name)
        vcache_name = "v_cache_{}".format(i)
        input_node(param, vcache_name, vcache_name)

    idx = 1
    weight_offset = 0

    #输入embedding
    idx, weight_offset = embed_node(param, "input_embed", "in", hidden_size, vocab_size, idx, weight_offset)

    #多层attention
    kv_cache_out = []
    for i in range(num_hidden_layers):
        idx, weight_offset, kcache_out, vcache_out = transformer_layer(model, param, i, idx, weight_offset)
        kv_cache_out.append([kcache_out, vcache_out])

    #输出norm
    idx_output_norm, weight_offset = output_norm(model, param, idx, weight_offset)

    #输出embedding
    idx_output_embed_weight, weight_offset = memorydata2_node(param, "output_embed_weight", vocab_size, hidden_size, weight_offset, idx_output_norm)
    matmul_node(param, "output_matmul", idx_output_norm, idx_output_embed_weight)

    kvcache_step = kv_cache_out[1][0] - kv_cache_out[0][0]
    param[1] = "{} {} {} {} {} {} {}".format(model.config.max_position_embeddings, model.config.num_hidden_layers, model.config.hidden_size, model.config.num_attention_heads, kv_cache_out[0][0], kv_cache_out[0][1], kvcache_step)

    output_node = param[-1].split(" ")
    output_node[-1] = "output"
    param[-1] = " ".join(output_node)


def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def export_model(model_path, param_file_path, bin_file_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code = True)

    print(model)

    param = []
    model_param(model, param)

    param_file = open(param_file_path, 'w')
    for i in range(len(param)):
        p = param[i]
        if i < len(param)-1:
            param_file.write(p+"\n")
        else:
            param_file.write(p)
    param_file.close()

    bin_file = open(bin_file_path, 'wb')

    model_attention = model.model
    lm_head = model.lm_head

    #输入embedding
    serialize_fp32(bin_file, model_attention.embed_tokens.weight)

    #多层attention
    for i in range(model.config.num_hidden_layers):
        serialize_fp32(bin_file, model_attention.layers[i].input_layernorm.weight)
        serialize_fp32(bin_file, model_attention.layers[i].self_attn.q_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].self_attn.k_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].self_attn.v_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].self_attn.o_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].post_attention_layernorm.weight)
        serialize_fp32(bin_file, model_attention.layers[i].mlp.gate_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].mlp.up_proj.weight)
        serialize_fp32(bin_file, model_attention.layers[i].mlp.down_proj.weight)

    #输出rms norm
    serialize_fp32(bin_file, model_attention.norm.weight)

    #输出embedding
    serialize_fp32(bin_file, lm_head.weight)

    bin_file.close()

if __name__ == "__main__":
    export_model('../models/chinese-baby-llama2', './test.ncnn.param', './test.ncnn.bin')


