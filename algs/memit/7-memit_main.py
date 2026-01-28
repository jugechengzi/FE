"""
MEMIT with Per-Layer Z and Hidden State Extraction (7-memit_main.py)
Use to extract hidden states under EDIT-ALL_LAYER method.

This file is built by combining:
- 5-memit_main.py: z_method=all, per-layer target z cached on disk
- 4-memit_main.py: hidden state extraction before/after each layer edit
"""

import os
import random
import time
from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from util.utility import ensure_file_directory
from util.h_cache_manager import save_h_batch, print_h_cache_status

from .compute_ks import compute_ks
from .compute_z import get_module_input_output_at_words, find_fact_lookup_idx


# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
covs: List[torch.Tensor] = []


def set_random_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_cov(cfg: DictConfig, model: AutoModelForCausalLM, tok: AutoTokenizer) -> None:
    """Load covariance matrices for all layers (K0K0T / second moment stats)."""
    global covs
    covs = []

    layers = cfg.llms.layers
    for layer in layers:
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
        if cfg.cov_mode == "random":
            print("Using random covariance matrix!")
            cov = torch.randn_like(cov)
        if cfg.cov_mode == "identity":
            print("Using identity covariance matrix!")
            cov = torch.eye(cov.shape[0])
        covs.append(cov)


def chunks(arr, n: int):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def get_fc_dim(model: AutoModelForCausalLM, cfg: DictConfig) -> int:
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim = W_out.shape[0] if W_out.shape[0] > W_out.shape[1] else W_out.shape[1]
    return int(fc_dim)


def extract_hidden_states(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig,
    layers: List[int],
    device: torch.device,
) -> Dict[int, torch.Tensor]:
    """
    Extract hidden states from specified layers for all requests.

    Returns:
        Dict mapping layer_num -> hidden_states tensor of shape [hidden_dim, num_samples]
    """
    hidden_states: Dict[int, List[torch.Tensor]] = {layer: [] for layer in layers}

    for request in requests:
        prompt = request["prompt"].format(request["subject"])
        input_tok = tok(prompt, return_tensors="pt").to(device)

        lookup_idx = find_fact_lookup_idx(
            request["prompt"],
            request["subject"],
            tok,
            cfg.llms.fact_token,
            verbose=False,
        )

        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[cfg.llms.layer_module_tmp.format(layer) for layer in layers],
            ) as tr:
                model(**input_tok)

            for layer in layers:
                layer_module = cfg.llms.layer_module_tmp.format(layer)
                output = tr[layer_module].output
                h_raw = output[0] if isinstance(output, tuple) else output

                if h_raw.dim() == 3:
                    # [batch, seq, hidden] or [seq, batch, hidden]
                    if h_raw.shape[0] == 1:
                        h = h_raw[0, lookup_idx, :]
                    else:
                        h = h_raw[lookup_idx, 0, :]
                else:
                    h = h_raw[lookup_idx, :]

                hidden_states[layer].append(h.detach().cpu())

    result: Dict[int, torch.Tensor] = {}
    for layer in layers:
        result[layer] = torch.stack(hidden_states[layer], dim=1)

    return result


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig,
):
    """Apply MEMIT (per-layer target z_method=all) and extract per-layer hidden states."""

    global CONTEXT_TEMPLATES_CACHE

    set_random_seed(int(getattr(cfg, "seed", 0)))
    model.eval()

    # Reset per-run caches to avoid cross-run contamination.
    CONTEXT_TEMPLATES_CACHE = None
    covs.clear()

    device = torch.device(f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu")

    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]

    layers = cfg.llms.layers
    last_layer = layers[-1]

    # Ensure K0K0T cache exists (will be loaded by get_cov)
    for layer in layers:
        cpath = (
            cfg.cache_dir
            + "/stats/"
            + cfg.llms.name.replace("/", "-")
            + "/layer-"
            + str(layer)
            + ("-" if cfg.cache_filename_suffix != "" else "")
            + cfg.cache_filename_suffix
            + ".npz"
        )
        ensure_file_directory(cpath)
        if not os.path.exists(cpath):
            print(
                f"The key matrix of old memory K0K0T for model {cfg.llms.name} layer {layer} "
                "does not exist and now calculate."
            )
            _ = get_cov(
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

    # Load full z cache once for all batches (per-layer, z_method=all)
    print("\nLoading full z cache for all layers (z_method=all)...")
    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = getattr(cfg, "data", "unknown")
    seed_value = getattr(cfg, "seed", 0)
    z_method = "all"

    zs_all: Dict[int, Optional[torch.Tensor]] = {layer: None for layer in layers}

    for layer in layers:
        cache_zs_file = (
            f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{layer}.pt"
        )
        if not os.path.isfile(cache_zs_file):
            raise FileNotFoundError(
                f"Cache file not found: {cache_zs_file}. "
                "Please ensure the cache file exists before running."
            )

        zs_full = torch.load(cache_zs_file, map_location="cpu")
        print(f"Loaded full z cache from {cache_zs_file}, shape: {zs_full.shape}")
        zs_all[layer] = zs_full

        num_cached_samples = zs_full.shape[1]
        num_required_samples = len(requests)
        if num_cached_samples < num_required_samples:
            raise ValueError(
                "Insufficient cached z samples! "
                f"Layer {layer}, Required: {num_required_samples}, Cached: {num_cached_samples}. "
                "Please pre-compute z for at least the required samples using precompute_z.py"
            )

    # Hidden state storage across batches
    h_pre_current_all = {layer: [] for layer in layers}
    h_pre_last_at_layer_all = {layer: [] for layer in layers}
    h_post_current_all = {layer: [] for layer in layers}
    h_post_last_at_layer_all = {layer: [] for layer in layers}

    # Like 4-memit: always extract hidden states from the completely unedited model.
    print("\n" + "=" * 80)
    print("EXTRACTING HIDDEN STATES FROM ORIGINAL MODEL (BEFORE ANY EDITS)")
    print("=" * 80)
    h_pre_original_all = {layer: [] for layer in layers}
    for requests_chunks in chunks(requests, cfg.bs):
        h_original_batch = extract_hidden_states(model, tok, requests_chunks, cfg, layers, device)
        for layer in layers:
            h_pre_original_all[layer].append(h_original_batch[layer])
    for layer in layers:
        h_pre_original_all[layer] = torch.cat(h_pre_original_all[layer], dim=1)
        print(f"  h_pre_original[{layer}] shape: {h_pre_original_all[layer].shape}")

    print("\nProcessing batches for layer-by-layer editing + hidden state extraction...")
    for requests_chunks in chunks(requests, cfg.bs):
        batch_results = batch_edit_layerwise(
            cfg,
            model,
            tok,
            requests_chunks,
            device,
            cache_c,
            zs_all,
            layers,
            last_layer,
        )
        for layer in layers:
            h_pre_current_all[layer].append(batch_results["h_pre_current"][layer])
            h_pre_last_at_layer_all[layer].append(batch_results["h_pre_last_at_layer"][layer])
            h_post_current_all[layer].append(batch_results["h_post_current"][layer])
            h_post_last_at_layer_all[layer].append(batch_results["h_post_last_at_layer"][layer])

    # Concatenate all batches
    for layer in layers:
        h_pre_current_all[layer] = torch.cat(h_pre_current_all[layer], dim=1)
        h_pre_last_at_layer_all[layer] = torch.cat(h_pre_last_at_layer_all[layer], dim=1)
        h_post_current_all[layer] = torch.cat(h_post_current_all[layer], dim=1)
        h_post_last_at_layer_all[layer] = torch.cat(h_post_last_at_layer_all[layer], dim=1)

    # Save hidden states
    h_cache_dir = cfg.get("h_cache_dir", "./h_cache")
    os.makedirs(h_cache_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING HIDDEN STATES")
    print("=" * 80)

    for layer in layers:
        save_h_batch(
            h_cache_dir,
            dataset_name,
            model_cache_name,
            "h_pre_original",
            h_pre_original_all[layer],
            layer=layer,
            seed=seed_value,
            append=False,
        )

    for layer in layers:
        save_h_batch(
            h_cache_dir,
            dataset_name,
            model_cache_name,
            "h_pre_current",
            h_pre_current_all[layer],
            layer=layer,
            seed=seed_value,
            append=False,
        )
        save_h_batch(
            h_cache_dir,
            dataset_name,
            model_cache_name,
            "h_pre_last_at_layer",
            h_pre_last_at_layer_all[layer],
            layer=layer,
            seed=seed_value,
            append=False,
        )
        save_h_batch(
            h_cache_dir,
            dataset_name,
            model_cache_name,
            "h_post_current",
            h_post_current_all[layer],
            layer=layer,
            seed=seed_value,
            append=False,
        )
        save_h_batch(
            h_cache_dir,
            dataset_name,
            model_cache_name,
            "h_post_last_at_layer",
            h_post_last_at_layer_all[layer],
            layer=layer,
            seed=seed_value,
            append=False,
        )

    print_h_cache_status(h_cache_dir, dataset_name, model_cache_name)

    return model


def batch_edit_layerwise(
    cfg: DictConfig,
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    device: torch.device,
    cache_c: torch.Tensor,
    zs_all: Dict[int, torch.Tensor],
    layers: List[int],
    last_layer: int,
) -> Dict[str, Dict[int, torch.Tensor]]:
    """Edit model weights layer-by-layer and extract hidden states before/after each layer edit."""

    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in layers
    }

    context_templates = get_context_templates(model, tok)

    # Extract z for current batch by sample indices
    sample_indices = [req["sample_idx"] for req in requests]
    max_cached_idx = max(zs_all[layer].shape[1] for layer in zs_all) - 1
    if max(sample_indices) > max_cached_idx:
        raise IndexError(
            f"Sample index out of bounds! Requested max index: {max(sample_indices)}, "
            f"Max cached index: {max_cached_idx}."
        )

    zs = {layer: zs_all[layer][:, sample_indices].to(device) for layer in layers}

    result: Dict[str, Dict[int, torch.Tensor]] = {
        "h_pre_current": {},
        "h_pre_last_at_layer": {},
        "h_post_current": {},
        "h_post_last_at_layer": {},
    }

    for i, layer in enumerate(layers):
        print("\n" + "=" * 80)
        print(f"PROCESSING LAYER {layer} ({i + 1}/{len(layers)})")
        print("=" * 80)

        # Extract hidden states BEFORE edit
        extraction_layers = [layer, last_layer] if layer != last_layer else [layer]
        h_before = extract_hidden_states(model, tok, requests, cfg, extraction_layers, device)
        result["h_pre_current"][layer] = h_before[layer]
        result["h_pre_last_at_layer"][layer] = h_before[last_layer]

        # Compute current keys
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Current z at this layer
        if cfg.negetive_prompt_test:
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
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                layer,
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T

        # Per-layer target z (z_method=all)
        targets = zs[layer] - cur_zs  # [dim, bs]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = layer_ks.size(1) // targets.size(1)
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        layer_ks = layer_ks.double()
        targets = targets.double()
        resid = targets

        cov = covs[i].double()
        coef = cfg.llms.mom2_update_weight[i]

        start_time = time.time()
        upd_matrix = torch.linalg.solve(
            layer_ks @ layer_ks.T
            + cache_c[i, :, :].to(device).double()
            + coef * cov.to(device)
            + cfg.algs.L2 * torch.eye(layer_ks.shape[0], device=device).double(),
            layer_ks @ resid.T,
        )
        end_time = time.time()
        print(f"Solved for update matrix in {end_time - start_time:.2f} seconds")

        if cfg.algs.add_old_keys:
            cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()

        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix

        # Extract hidden states AFTER edit
        h_after = extract_hidden_states(model, tok, requests, cfg, extraction_layers, device)
        result["h_post_current"][layer] = h_after[layer]
        result["h_post_last_at_layer"][layer] = h_after[last_layer]

        h_change = torch.linalg.norm(
            result["h_post_current"][layer] - result["h_pre_current"][layer], dim=0
        ).mean()
        last_change = torch.linalg.norm(
            result["h_post_last_at_layer"][layer] - result["h_pre_last_at_layer"][layer], dim=0
        ).mean()
        print(f"Δh_current[{layer}]: {h_change:.4f}")
        print(f"Δh_last[{last_layer}] after editing layer {layer}: {last_change:.4f}")

    return result


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """Match update matrix to weight shape (handles GPT-2/GPT-J transpose conventions)."""
    if matrix.shape == shape:
        return matrix
    if matrix.T.shape == shape:
        return matrix.T
    raise ValueError(
        "Update matrix computed by MEMIT does not match original weight shape. "
        "Check for bugs in the code?"
    )


def get_context_templates(model: AutoModelForCausalLM, tok: AutoTokenizer):
    """Get context templates for z computation (must match precompute_z.py)."""
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
