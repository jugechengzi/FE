"""
NAMET (variant): use cached z (z_method=all) from last edit layer.
Unfortunately this version is not correct. 
NAMET sets the tok padding side to "left", and the compute_z.py function has also been modified to handle left padding as well. However, the cached zs are still computed with right padding, which causes a mismatch between the cached zs and the actual module inputs during editing.
"""
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
covs = []  # 将K0K0T先从文件读取到cpu上，之后不用再读文件


def load_cov(cfg, model, tok):
    layers = cfg.llms.layers
    for i, layer in enumerate(layers):
        cov = get_cov(
            cfg,
            model,
            tok,
            layer,
            cfg.llms.mom2_dataset,
            cfg.llms.mom2_n_samples,
            cfg.llms.mom2_dtype,
            force_recompute=False,
            cache_filename_suffix=cfg.cache_filename_suffix,
        )
        covs.append(cov)


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def get_fc_dim(model, cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim = W_out.shape[0] if W_out.shape[0] > W_out.shape[1] else W_out.shape[1]
    return fc_dim


def apply_namet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig,
):
    """NAMET (variant): use cached z (z_method=all) from last edit layer."""

    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]

    layers = cfg.llms.layers
    for i, layer in enumerate(layers):
        Cpathi = (
            cfg.cache_dir
            + "/stats/"
            + cfg.llms.name.replace("/", "-")
            + "/layer-"
            + str(layer)
            + ("-" if cfg.cache_filename_suffix != "" else "")
            + cfg.cache_filename_suffix
            + ".npz"
        )
        ensure_file_directory(Cpathi)
        if not os.path.exists(Cpathi):
            print(
                "The key matrix of old memory K0K0T for model {} layer {} "
                "does not exist and now calculate.".format(cfg.llms.name, layer)
            )
            get_cov(
                cfg,
                model,
                tok,
                layer,
                cfg.llms.mom2_dataset,
                cfg.llms.mom2_n_samples,
                cfg.llms.mom2_dtype,
                force_recompute=False,
                cache_filename_suffix=cfg.cache_filename_suffix,
            )

    load_cov(cfg, model, tok)
    fc_dim = get_fc_dim(model, cfg)
    cache_c = torch.zeros((len(layers), fc_dim, fc_dim), device="cpu")

    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = getattr(cfg, "data", "unknown")
    seed_value = getattr(cfg, "seed", 0)
    z_method = "all" # use precomputed z for last edit layer
    z_layer = cfg.llms.layers[-1]
    cache_zs_file = f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{z_layer}.pt"
    if not os.path.isfile(cache_zs_file):
        raise FileNotFoundError(
            f"Cache file not found: {cache_zs_file}. Please pre-compute with: "
            f"python precompute_z.py --edited_layers={','.join(map(str, cfg.llms.layers))} z_method=all num_z_samples=..."
        )
    zs_full = torch.load(cache_zs_file, map_location="cpu")
    print(f"Loaded z cache from {cache_zs_file}, shape: {zs_full.shape}")
    if zs_full.shape[1] < len(requests):
        raise ValueError(
            f"Insufficient cached z samples! Required: {len(requests)}, Cached: {zs_full.shape[1]}. "
            f"Please pre-compute z for at least {len(requests)} samples using precompute_z.py"
        )

    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg, model, tok, requests_chunks, device, cache_c, zs_full, z_layer)
    return model


def batch_edit(cfg, model, tok, requests, device, cache_c, zs_full, z_layer):
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }

    context_templates = get_context_templates(model, tok)

    sample_indices = [req["sample_idx"] for req in requests]
    max_cached_idx = zs_full.shape[1] - 1
    max_requested_idx = max(sample_indices)
    if max_requested_idx > max_cached_idx:
        raise IndexError(
            f"Sample index out of bounds! Requested index: {max_requested_idx}, Max cached index: {max_cached_idx}. "
            f"Cache shape: {zs_full.shape}"
        )
    zs = zs_full[:, sample_indices].to(device)

    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        if cfg.negetive_prompt_test:
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

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(cfg.llms.layers) - i)

        cov = covs[i]
        upd_type = torch.float
        coef = cfg.llms.mom2_update_weight[i]

        if cfg.algs.L2 != 0:
            upd_matrix = torch.linalg.solve(
                layer_ks @ layer_ks.T
                + cache_c[i, :, :].to(device)
                + cfg.algs.L2 * torch.eye(layer_ks.shape[0], dtype=upd_type, device=device),
                layer_ks.to(upd_type) @ resid.T,
            )
        else:
            upd_matrix = torch.linalg.solve(
                layer_ks @ layer_ks.T
                + cache_c[i, :, :].to(device)
                + coef * cov.to(device)
                + cfg.algs.L2 * torch.eye(layer_ks.shape[0], dtype=upd_type, device=device),
                layer_ks.to(upd_type) @ resid.T,
            )

        if cfg.algs.add_old_keys:
            cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()

        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
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
            for length, n_gen in [(10, 5)]
        ]
        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
