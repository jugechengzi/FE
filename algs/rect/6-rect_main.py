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
            cache_filename_suffix=cfg.cache_filename_suffix
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

def rect_norm(cfg,upd_matrix):
    flattened_matrix = upd_matrix.flatten()
    flattened_pos_matrix=torch.abs(upd_matrix).flatten()
    num_elements = flattened_pos_matrix.nelement()
    num_zeros = int(num_elements * cfg.algs.zero_prob)
    indices_to_zero = torch.topk(flattened_pos_matrix, k=num_zeros, largest=False)[1]#1是索引。
    flattened_matrix[indices_to_zero] = 0
    upd_matrix = flattened_matrix.reshape(upd_matrix.shape)
    return upd_matrix

def apply_rect_to_model(
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
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    #查看KKT是否已经计算好。
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +("-" if cfg.cache_filename_suffix !="" else "")+ cfg.cache_filename_suffix + ".npz"
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
                cache_filename_suffix=cfg.cache_filename_suffix
            )
            #这个内部会自动保存，我们不需要再额外管。
    load_cov(cfg,model,tok)
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

def batch_edit(cfg, model, tok, requests, device, cache_c, zs_all):
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
                layer, # layer number, not final layer
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs[layer] - cur_zs#[dim,bs]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets  # Do not distribute residual across layers

        cov = covs[i]
        upd_type = torch.float32
        ks = layer_ks.to(dtype=upd_type)
        resid = resid.to(dtype=upd_type)
        cache = cache_c[i, :, :].to(device=device, dtype=upd_type)
        cov_upd = cov.to(device=device, dtype=upd_type)

        if cfg.algs.L2 != 0:
            upd_matrix = torch.linalg.solve(
                ks @ ks.T + cache
                + cfg.algs.L2 * torch.eye(ks.shape[0], dtype=upd_type, device=device),
                ks @ resid.T,
            )
        else:
            coef=cfg.llms.mom2_update_weight[i]
            upd_matrix = torch.linalg.solve(
                ks @ ks.T + cache + coef * cov_upd
                + cfg.algs.L2 * torch.eye(ks.shape[0], dtype=upd_type, device=device),
                ks @ resid.T,
            )
        upd_matrix=rect_norm(cfg,upd_matrix)
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
