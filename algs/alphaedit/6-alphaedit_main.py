"""
AlphaEdit BLUE implementation with precomputed Z cache support. (6-alphaedit_main.py)
"""
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
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +("-" if cfg.cache_filename_suffix!="" else "")+ cfg.cache_filename_suffix + ".pt"
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
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    # compute the null space project P.
    for i, layer in enumerate(layers):
        Ppathi = cfg.cache_dir + "/null_space_project/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +("-" if cfg.cache_filename_suffix!="" else "")+ cfg.cache_filename_suffix + ".pt"
        ensure_file_directory(Ppathi)
        if not os.path.exists(Ppathi):#then compute
            print("The null-space projection matrix P for model {} layer {} "
                  "does not exist and now calculate.".format(cfg.llms.name.replace("/","-"), layer))
            Pi = get_project(model, tok, layer, cfg)  # 对于每一个选定的层，都要计算P，也是牛逼了，我觉得到时候我们不要计算这么多层， 可以节约时间。
            torch.save(Pi.cpu(), Ppathi)
    load_project(cfg)
    fc_dim=get_fc_dim(model,cfg)
    cache_c = torch.zeros((len(layers), fc_dim,fc_dim), device="cpu")

    # Load z cache once for all batches (enables flexible batch size)
    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = getattr(cfg, 'data', 'unknown')
    seed_value = getattr(cfg, 'seed', 0)
    z_method = "all"  # Per-layer z targets

    zs_all = {layer: None for layer in cfg.llms.layers}
    for layer in cfg.llms.layers:
        cache_zs_file = f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{layer}.pt"
        if os.path.isfile(cache_zs_file):
            zs_full = torch.load(cache_zs_file, map_location='cpu')
            print(f"Loaded full z cache from {cache_zs_file}, shape: {zs_full.shape}")
            zs_all[layer] = zs_full
            num_cached_samples = zs_full.shape[1]
            num_required_samples = len(requests)
            if num_cached_samples < num_required_samples:
                raise ValueError(
                    f"Insufficient cached z samples! "
                    f"Required: {num_required_samples}, Cached: {num_cached_samples}. "
                    f"Please pre-compute z for at least {num_required_samples} samples using precompute_z.py"
                )
        else:
            raise FileNotFoundError(
                f"Cache file not found: {cache_zs_file}. Please ensure the cache file exists before running."
            )

    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg,model,tok,requests_chunks,device,cache_c,zs_all)
    return model


def batch_edit(cfg,model,tok,requests,device,cache_c,zs_all):
    # deltas = {}
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    context_templates = get_context_templates(model, tok)

    sample_indices = [req["sample_idx"] for req in requests]
    max_cached_idx = max(zs_all[layer].shape[1] for layer in zs_all) - 1
    max_requested_idx = max(sample_indices)
    if max_requested_idx > max_cached_idx:
        raise IndexError(
            f"Sample index out of bounds! "
            f"Requested index: {max_requested_idx}, Max cached index: {max_cached_idx}. "
            f"Cache shapes: {[zs_all[layer].shape for layer in zs_all]}"
        )

    zs = {layer: zs_all[layer][:, sample_indices].to(device) for layer in zs_all}

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
                layer,
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
                layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs[layer] - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets  # Do not distribute residual across layers

        # Keep linear algebra dtypes consistent (cached z / activations may be bf16).
        upd_type = torch.float32
        Pi_upd = Pi.to(dtype=upd_type)
        ks = layer_ks.to(dtype=upd_type)
        resid = resid.to(dtype=upd_type)
        cache = cache_c[i, :, :].to(device=device, dtype=upd_type)

        system = Pi_upd @ (ks @ ks.T + cache) + cfg.algs.L2 * torch.eye(
            ks.shape[0], dtype=upd_type, device=device
        )
        rhs = Pi_upd @ ks @ resid.T
        upd_matrix = torch.linalg.solve(system, rhs)
        #用完了cache_c[i]之后更新旧记忆。
        if cfg.algs.add_old_keys:
            cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()

        # Adjust update matrix shape
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix.to(dtype=weights[weight_name].dtype)
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
