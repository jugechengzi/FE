import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import hydra
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))          # 当前文件目录
parent_dir = os.path.dirname(current_dir)                         # 父目录
sys.path.append(parent_dir)

from load import load_model

def get_attention_scores(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # 关键：输出 attentions
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            use_cache=False,  # 必须，否则部分层可能不返回 attention
        )

    attentions = outputs.attentions   # list, len = num_layers
    # 对于 llama3-8b，num_layers = 32

    # 获取每层最后一个 token 的注意力
    last_token_att_per_layer = []

    for layer_idx, attn in enumerate(attentions):
        # attn: [batch, num_heads, seq_len, seq_len]
        # 我们取 batch=0, token=-1
        last_token_att = attn[0, :, -1, :].mean(dim=0)  # [seq_len]
        last_token_att_per_layer.append(last_token_att[1:])  # 去掉 begin_token 的注意力

    # 转成 tensor: [num_layers, seq_len]
    layer_token_att = torch.stack(last_token_att_per_layer)
    return layer_token_att

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to("cuda")
    model = load_model(model, cfg)
    text = "Apple A5 was created by"
    layer_token_att_per_layer = get_attention_scores(model, tokenizer, text)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.imshow(layer_token_att_per_layer.T.detach().cpu(), aspect='auto')
    plt.colorbar(label="Attention score")
    plt.xlabel("Layer index")
    plt.ylabel("Token index")
    plt.title(f'"{text}" Attention Score Heatmap')
    plt.savefig("attention_heatmap_2_AppleA5.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()