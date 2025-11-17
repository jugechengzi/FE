import hydra
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from load import load_model
from util.nethook import Trace

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
    ).to("cuda")
    text = "Apple A5 was created by "
    negetive_text = "Apple A5 was not created by "
    print(f"Text: {text}\nNegetive Text: {negetive_text}")

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    negetive_text_inputs = tokenizer(negetive_text, return_tensors="pt").to(model.device)

    with Trace(
            model, cfg.llms.rewrite_module_tmp.format(8), retain_input=False, retain_output=True, stop=True
    ) as tr:
        model(**inputs)
    m0 = tr.output[0].detach()

    edited_model = load_model(model, cfg)
    with Trace(
            edited_model, cfg.llms.rewrite_module_tmp.format(8), retain_input=False, retain_output=True, stop=True
    ) as tr:
        edited_model(**inputs)
    m1 = tr.output[0].detach()
    with Trace(
            model, cfg.llms.rewrite_module_tmp.format(8), retain_input=False, retain_output=True, stop=True
    ) as tr:
        model(**negetive_text_inputs)
    m2 = tr.output[0].detach()

    m0_last_subject_token = m0[3:4]
    m1_last_subject_token = m1[3:4]
    m2_last_subject_token = m2[3:4]
    cosine_similarity_m0_m1 = F.cosine_similarity(m0_last_subject_token, m1_last_subject_token)
    cosine_similarity_m1_m2 = F.cosine_similarity(m1_last_subject_token, m2_last_subject_token)
    print("Cosine Similarity between original and edited model outputs (last subject token m0 vs m1):")
    print(cosine_similarity_m0_m1)
    print("Cosine Similarity between edited model outputs for positive and negative texts (last subject token m1 vs m2):")
    print(cosine_similarity_m1_m2)
    diff_norm_m0_m1 = torch.norm(m0_last_subject_token - m1_last_subject_token, p=2)
    diff_norm_m1_m2 = torch.norm(m1_last_subject_token - m2_last_subject_token, p=2)
    print("L2 Norm of difference between original and edited model outputs (last subject token m0 vs m1):")
    print(diff_norm_m0_m1)
    print("L2 Norm of difference between edited model outputs for positive and negative texts (last subject token m1 vs m2):") 
    print(diff_norm_m1_m2)

    m0_last_token = m0[-1:]
    m1_last_token = m1[-1:]
    m2_last_token = m2[-1:]
    cosine_similarity_last_m0_m1 = F.cosine_similarity(m0_last_token, m1_last_token)
    cosine_similarity_last_m1_m2 = F.cosine_similarity(m1_last_token, m2_last_token)
    print("Cosine Similarity between original and edited model outputs (last token m0 vs m1):")
    print(cosine_similarity_last_m0_m1)
    print("Cosine Similarity between edited model outputs for positive and negative texts (last token m1 vs m2):")
    print(cosine_similarity_last_m1_m2)
    diff_norm_last_m0_m1 = torch.norm(m0_last_token - m1_last_token, p=2)
    diff_norm_last_m1_m2 = torch.norm(m1_last_token - m2_last_token, p=2)
    print("L2 Norm of difference between original and edited model outputs (last token m0 vs m1):")
    print(diff_norm_last_m0_m1)
    print("L2 Norm of difference between edited model outputs for positive and negative texts (last token m1 vs m2):")
    print(diff_norm_last_m1_m2)


if __name__ == "__main__":
    main()