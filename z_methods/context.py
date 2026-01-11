from transformers import AutoModelForCausalLM, AutoTokenizer
from random import shuffle
from copy import deepcopy
from typing import Dict

from util import nethook
import torch


def get_edit_prefix(
    request: Dict
) -> str:
    new_fact = request['prompt'].format(
        request['subject']) + request['target_new']

    context_demo = f"Imagine that {new_fact}\n"

    return context_demo


def get_edit_target(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    cfg,
    context_prefix: str,
    rewriting_prompts: list,
    loc_prompts: list,
    target_idxs_start: list,
    target_idxs_end: list,
    lookup_idxs: list,
):

    print(context_prefix)

    rewriting_prompts = [context_prefix+prompt for prompt in rewriting_prompts]
    all_prompts = rewriting_prompts + loc_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    lm_w, ln_f = (
        nethook.get_parameter(model, f"{cfg.lm_head_module}.weight").T,
        nethook.get_module(model, cfg.ln_f_module),
    )
    lm_b = nethook.get_parameter(model, f"{cfg.lm_head_module}.bias")
    if lm_b is None:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    trace_layers_mid = [
        cfg.layer_module_tmp.format(layer) for layer in cfg.midlayers
    ]
    trace_layers_final = [
        cfg.layer_module_tmp.format(layer+1) for layer in cfg.midlayers
    ]

    midlayer_vec, subject_vec, last_vec = None, None, None
    out_tuple = None
    def get_output_fn(cur_out, cur_layer):
        nonlocal midlayer_vec, subject_vec, last_vec
        nonlocal out_tuple
        out_tuple=type(cur_out)==tuple
        if not out_tuple:#正常情况下，cur_out[0]是该层输出，cur_out[1]是该层注意力权重。
            cur_out = (cur_out,)#但是有一些大模型例如llama3-8b不输出注意力权重，只有该层输出，是一个tensor。
        if cur_layer in trace_layers_mid:
            tmp_repr = cur_out[0]
            subject_vec = torch.stack(
                [
                    tmp_repr[i, idx, :]
                    for i, idx in enumerate(lookup_idxs)
                ],
                dim=0
            ).unsqueeze(1)

        if cur_layer in trace_layers_final:
            tmp_repr = cur_out[0]
            last_vec = torch.cat(
                [
                    tmp_repr[i, -idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0
            ).unsqueeze(1)

        # 根据 cfg.constr_pos 决定返回哪个向量或组合
        if cfg.constr_pos == "subject":
            midlayer_vec = subject_vec
        elif cfg.constr_pos == "last":
            midlayer_vec = last_vec
        elif cfg.constr_pos == "all":
            if last_vec is not None:
                midlayer_vec = torch.cat([subject_vec, last_vec], dim=0)
        else:
            raise ValueError(
                f"Unsupported constr_pos: {cfg.constr_pos}")

        return cur_out

    with torch.no_grad():
        with nethook.TraceDict(
            module=model,
            layers=trace_layers_mid + trace_layers_final,
            retain_input=False,
            retain_output=True,
            edit_output=get_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            kl_logits = torch.cat(
                [
                    logits[i, idxst:idxed, :]
                    for i, (idxst, idxed) in enumerate(zip(target_idxs_start, target_idxs_end))
                ],
                dim=0,
            )

            print(f"MIDLAYER_VEC_SHAPE:{midlayer_vec.shape}")
            midlayer_logits = ln_f(
                midlayer_vec) @ lm_w.to(midlayer_vec.device) + lm_b.to(midlayer_vec.device)

            print(target_idxs_start)
            print(target_idxs_end)
            print(f"TARGETSHAPE:{kl_logits.shape}")

            mid_kl_log_probs = torch.nn.functional.log_softmax(
                midlayer_logits.squeeze(1), dim=1).detach().clone()
            target_kl_log_probs = torch.nn.functional.log_softmax(
                kl_logits, dim=1).detach().clone()

    return target_kl_log_probs, mid_kl_log_probs
