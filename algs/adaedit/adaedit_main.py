import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from util.utility import ensure_file_directory

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from omegaconf import DictConfig

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
covs=[]#将K0K0T先从文件读取到cpu上，之后不用再读文件，可以显著加快速度（空间换时间），尤其是批次batch size小的时候。
def load_cov(cfg,model,tok):
    layers=cfg.llms.layers
    for i, layer in enumerate(layers):
        cov=get_cov(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
        )
        covs.append(cov)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

def topk_abs_keep(tensor, k,dim):
    _, indices = torch.topk(tensor.abs(), k=k,dim=dim)
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    mask.scatter_(dim, indices, True)
    result = tensor * mask
    return result

def norm_keep(tensor_org,tensor_new):
    factor=tensor_org.norm()/tensor_new.norm()
    tensor_new=tensor_new*factor
    return tensor_new

def adaedit_norm(cfg,upd_matrix):
    # edited_matrix=orig_matrix+upd_matrix
    u, s, v = torch.linalg.svd(upd_matrix, full_matrices=False)#原本是True，为了加速，我这里换成了False，按照原理应该是没有问题的，不改变结果，如果有问题，换回来即可。
    #第二个参数好像是针对非方阵的，我们这里好像恰好是非方阵，需要这个参数，会改变u,v的值。[6400,1600]的奇异值分解，设置为true，那么u的形状是4d,4d，设置为false，那么为4d,d，v的形状是d,d不变。s的形状是d不变。

    topr = int(len(s)*cfg.algs.topr)#想要多少个最大的奇异值和对应的奇异向量。

    ##new delta
    u1 = u[:, :topr]#更新矩阵的部分u向量。
    s1 = torch.diag(s[:topr])#更新矩阵的部分奇异值。
    v1 = v[:topr]#更新矩阵的部分v向量。

    #下面还要对u1,v1变形。
    topk_u = int(u1.shape[0]*cfg.algs.topk)#想要将奇异向量中多少个元素置为0.
    topk_v = int(v1.shape[1]*cfg.algs.topk)#想要将奇异向量中多少个元素置为0.

    u2=topk_abs_keep(u1, topk_u,dim=0)
    v2=topk_abs_keep(v1, topk_v,dim=1)#接下来还要norm。

    u3=norm_keep(u1,u2)
    v3=norm_keep(v1,v2)

    upd_matrix_norm = u3@s1@v3
    return upd_matrix_norm

def apply_adaedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    #查看KKT是否已经计算好。
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) + ".npz"
        ensure_file_directory(Cpathi)
        if not os.path.exists(Cpathi):#then compute
            print("The key matrix of old memory K0K0T for model {} layer {} "
                  "does not exist and now calculate.".format(cfg.llms.name, layer))
            cov = get_cov(
                cfg,
                model,
                tok,
                layer,
                cfg.llms.mom2_dataset,
                cfg.llms.mom2_n_samples,
                cfg.llms.mom2_dtype,
                force_recompute=False,
            )
            #这个内部会自动保存，我们不需要再额外管。
    load_cov(cfg,model,tok)
    fc_dim=get_fc_dim(model,cfg)
    cache_c = torch.zeros((len(layers), fc_dim,fc_dim), device="cpu")
    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg,model,tok,requests_chunks,device,cache_c)
    return model

def batch_edit(cfg, model, tok, requests, device, cache_c):
    # deltas = {}
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    z_layer = cfg.llms.layers[-1]
    z_list = []

    for request in requests:
        cur_z = compute_z(
            model,
            tok,
            request,
            cfg,
            z_layer,
            context_templates,
        )
        z_list.append(cur_z)
    zs = torch.stack(z_list, dim=1)#[dim,bs]

    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        # Get current model activations
        # layer_ks = compute_ks(model, tok, requests, cfg, cfg.llms.rewrite_module_tmp,layer, context_templates)
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        if cfg.negetive_prompt_test:
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["negetive_prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.rewrite_module_tmp, #######这里原来是层输出，现在变成了mlp输出。
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        else:
            # Compute residual error
            '''改动之处'''
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.rewrite_module_tmp, #######这里原来是层输出，现在变成了mlp输出。
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs - cur_zs#[dim,bs]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers

        cov = covs[i]
        upd_type = torch.float

        coef=cfg.llms.mom2_update_weight[i]
        upd_matrix = torch.linalg.solve(
            layer_ks @ layer_ks.T + cache_c[i, :, :].to(device)+coef*cov.to(device)+
            cfg.algs.L2 * torch.eye(layer_ks.shape[0], dtype=upd_type, device=device),
            layer_ks.to(upd_type) @ resid.T,
        )
        upd_matrix=adaedit_norm(cfg,upd_matrix)
        if cfg.algs.add_old_keys:
            cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()
        # Adjust update matrix shape
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
            # deltas[weight_name] = upd_matrix

        # cov.cpu()
        # for x in [layer_ks, cur_zs, targets]:
        #     x.cpu()
        #     del x
        # torch.cuda.empty_cache()
    #
    # if cfg.algs.add_old_keys:
    #     for i, layer in enumerate(cfg.llms.layers):
    #         layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
    #         cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    model,
                    tok,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
