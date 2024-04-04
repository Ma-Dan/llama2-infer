from contextlib import nullcontext
import torch
from model import ModelArgs, Transformer
import struct

torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def export_model(checkpoint, filepath):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint_dict = torch.load(checkpoint, map_location=device)
    gptconf = ModelArgs(**checkpoint_dict['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint_dict['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    print(model)

    out_file = open(filepath, 'wb')

    serialize_fp32(out_file, model.tok_embeddings.weight)

    out_file.close()

if __name__ == "__main__":
    export_model('../models/tinyllamas/stories42M.pt', './test.ncnn.bin')


