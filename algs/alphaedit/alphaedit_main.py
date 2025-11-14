import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from omegaconf import DictConfig
from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
# Cache variable(s)
Ps=[]#在cpu放置零空间矩阵（只需要一次读入），从而不需要每一个批次从文件中读取，后者更加耗时。
CONTEXT_TEMPLATES_CACHE = None
from util.utility import ensure_file_directory
def load_project(cfg):
    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +"-"+ cfg.cache_filename_suffix + ".pt"
        Pi = torch.load(Ppathi,map_location="cpu")  #这个矩阵通常是比较大的，比如1-2个G，多个层那么就多个G，先放在cpu上，然后按需放在gpu上。
        Ps.append(Pi)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

def apply_alphaedit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    # compute the null space project P.
    for i, layer in enumerate(layers):
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +"-"+ cfg.cache_filename_suffix + ".pt"
        ensure_file_directory(Ppathi)
        if not os.path.exists(Ppathi):#then compute
            print("The null-space projection matrix P for model {} layer {} "
                  "does not exist and now calculate.".format(cfg.llms.name.replace("/","-"), layer))
            Pi = get_project(model, tok, layer, cfg)  # 对于每一个选定的层，都要计算P，也是牛逼了，我觉得到时候我们不要计算这么多层， 可以节约时间。
            torch.save(Pi.cpu(), Ppathi)
    load_project(cfg)
    fc_dim=get_fc_dim(model,cfg)
    cache_c = torch.zeros((len(layers), fc_dim,fc_dim), device="cpu")
    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg,model,tok,requests_chunks,device,cache_c)
    return model


def batch_edit(cfg,model,tok,requests,device,cache_c):
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
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        Pi = Ps[i].to(device)  #节约gpu。
        # print(layer_ks.sum(),layer_ks.mean(),layer_ks.std())
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        if cfg.negetive_prompt_test:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["negetive_prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        else:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        upd_type=torch.float
        upd_matrix = torch.linalg.solve(
                Pi @ (layer_ks @ layer_ks.T + cache_c[i,:,:].to(device)) +
                cfg.algs.L2*torch.eye(layer_ks.shape[0], dtype=upd_type,device=device),
            Pi @ layer_ks.to(upd_type) @ resid.T
        )
        #用完了cache_c[i]之后更新旧记忆。
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

        # Pi.cpu()
        # for x in [layer_ks, cur_zs, targets]:
        #     x.cpu()
        #     del x
        # torch.cuda.empty_cache()

    # if cfg.algs.add_old_keys:
    #     for i, layer in enumerate(cfg.llms.layers):
    #         layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
    #         cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()

def update_model(model,deltas,recover=False):
    with torch.no_grad():
        for weight_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, weight_name)
            if recover:
                w[...] -= upd_matrix
            else:
                w[...] += upd_matrix
    return model#更新w，也就更新了model。

def get_project(model, tok, layer, cfg):
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    force_recompute = False
    cov = get_cov(
        cfg,
        model,
        tok,
        layer,
        cfg.llms.mom2_dataset,
        cfg.llms.mom2_n_samples,
        cfg.llms.mom2_dtype,
        force_recompute=force_recompute,
        cache_filename_suffix=cfg.cache_filename_suffix
    )
    U, S, _ = torch.linalg.svd(cov.to(device), full_matrices=False)
    threshold = cfg.algs.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print("the number of small singular values",len(small_singular_indices))
    return U[:, small_singular_indices] @ U[:, small_singular_indices].T


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
