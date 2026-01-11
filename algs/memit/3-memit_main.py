"""
MEMIT with Hidden State Extraction (3-memit_main.py)

This variant extracts hidden states before and after editing:
- h_pre^i: Output of layer i before editing
- h_pre^last: Output of last edit layer before editing
- h_post^i: Output of layer i after editing
- h_post^last: Output of last edit layer after editing

All hidden states are saved in format compatible with precompute_z.py for easy comparison.
"""

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.layer_stats import get_cov
from util import nethook
from util.generate import generate_fast
from util.utility import ensure_file_directory
from util.h_cache_manager import save_h_batch, print_h_cache_status

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from omegaconf import DictConfig

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
covs = []  # Store K0K0T covariance matrices


def load_cov(cfg, model, tok):
    """Load covariance matrices for all layers."""
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
            cache_filename_suffix=cfg.cache_filename_suffix
        )
        if cfg.cov_mode == "random":
            print("Using random covariance matrix!")
            cov = torch.randn_like(cov)
        if cfg.cov_mode == "identity":
            print("Using identity covariance matrix!")
            cov = torch.eye(cov.shape[0])
        covs.append(cov)


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


def get_fc_dim(model, cfg):
    """Get the fully connected layer dimension."""
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim = W_out.shape[0] if W_out.shape[0] > W_out.shape[1] else W_out.shape[1]
    return fc_dim


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
    hidden_states = {layer: [] for layer in layers}
    
    for request in requests:
        # Prepare input
        prompt = request["prompt"].format(request["subject"])
        input_tok = tok(prompt, return_tensors="pt").to(device)
        
        # Find lookup index (same as compute_z)
        fact_token = cfg.llms.fact_token
        lookup_idx = find_fact_lookup_idx(
            request["prompt"], 
            request["subject"], 
            tok, 
            fact_token, 
            verbose=False
        )
        
        # Extract hidden states using TraceDict
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[cfg.llms.layer_module_tmp.format(layer) for layer in layers],
            ) as tr:
                model(**input_tok)
            
            # Collect outputs from each layer
            for layer in layers:
                layer_module = cfg.llms.layer_module_tmp.format(layer)
                output = tr[layer_module].output
                
                # Extract hidden state at lookup position
                if isinstance(output, tuple):
                    h_raw = output[0]
                else:
                    h_raw = output
                
                # Handle different tensor shapes
                if h_raw.dim() == 3:
                    # [batch, seq, hidden] or [seq, batch, hidden]
                    if h_raw.shape[0] == 1:
                        h = h_raw[0, lookup_idx, :]
                    else:
                        h = h_raw[lookup_idx, 0, :]
                else:
                    # [seq, hidden]
                    h = h_raw[lookup_idx, :]
                
                hidden_states[layer].append(h.detach().cpu())
    
    # Stack into tensors [hidden_dim, num_samples]
    result = {}
    for layer in layers:
        result[layer] = torch.stack(hidden_states[layer], dim=1)
    
    return result


def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    cfg: DictConfig
):
    """
    Apply MEMIT to model and extract hidden states before/after editing.
    
    Returns:
        model: The updated model
    """
    weights_copy = {}
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    requests = deepcopy(requests)
    
    for i, request in enumerate(requests):
        # Add sample index if not already present (for z extraction)
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]
    
    layers = cfg.llms.layers
    last_layer = layers[-1]  # Last edit layer
    
    # Check and compute KKT if needed
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/" + cfg.llms.name.replace("/", "-") + "/layer-" + str(layer) + ("-" if cfg.cache_filename_suffix != "" else "") + cfg.cache_filename_suffix + ".npz"
        ensure_file_directory(Cpathi)
        if not os.path.exists(Cpathi):
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
    
    load_cov(cfg, model, tok)
    fc_dim = get_fc_dim(model, cfg)
    cache_c = torch.zeros((len(layers), fc_dim, fc_dim), device="cpu")
    
    # Load z cache
    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = getattr(cfg, 'data', 'unknown')
    seed_value = getattr(cfg, 'seed', 0)
    z_method = "all"
    z_layer = cfg.llms.layers[-1]
    
    print(f"\nLoading full z cache for MEMIT...")
    cache_zs_file = f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{z_layer}.pt"
    
    if os.path.isfile(cache_zs_file):
        zs_full = torch.load(cache_zs_file, map_location='cpu')
        print(f"Loaded full z cache from {cache_zs_file}, shape: {zs_full.shape}")
        
        num_cached_samples = zs_full.shape[1]
        num_required_samples = len(requests)
        if num_cached_samples < num_required_samples:
            raise ValueError(
                f"Insufficient cached z samples! "
                f"Required: {num_required_samples}, Cached: {num_cached_samples}. "
                f"Please pre-compute z for at least {num_required_samples} samples using precompute_z.py"
            )
        print(f"✓ Cache validation passed: {num_cached_samples} samples available, {num_required_samples} required")
    else:
        raise FileNotFoundError(f"Cache file not found: {cache_zs_file}. Please ensure the cache file exists before running.")
    
    # Initialize storage for layer-by-layer extraction
    # For each edit layer i: h_pre_i, h_pre_last_at_i, h_post_i, h_post_last_at_i
    h_pre_current_all = {layer: [] for layer in layers}  # h before editing layer i
    h_pre_last_at_layer = {layer: [] for layer in layers}  # last layer output before editing layer i
    h_post_current_all = {layer: [] for layer in layers}  # h after editing layer i
    h_post_last_at_layer = {layer: [] for layer in layers}  # last layer output after editing layer i
    
    # NEW: Extract h from ORIGINAL model (before any edits) for cumulative effect analysis
    print(f"\n{'='*80}")
    print(f"EXTRACTING HIDDEN STATES FROM ORIGINAL MODEL (BEFORE ANY EDITS)")
    print(f"{'='*80}")
    h_pre_original_all = {layer: [] for layer in layers}  # Original model output for each layer
    
    # Extract original model outputs for ALL samples BEFORE any editing
    print(f"\nExtracting h_pre_original from completely unedited model for all samples...")
    for requests_chunks in chunks(requests, cfg.bs):
        h_original_batch = extract_hidden_states(
            model, tok, requests_chunks, cfg, layers, device
        )
        for layer in layers:
            h_pre_original_all[layer].append(h_original_batch[layer])
    
    # Concatenate all original outputs
    for layer in layers:
        h_pre_original_all[layer] = torch.cat(h_pre_original_all[layer], dim=1)
        print(f"  h_pre_original[{layer}] shape: {h_pre_original_all[layer].shape}")
    
    # Process in batches for editing
    print(f"\nProcessing batches for layer-by-layer editing...")
    for requests_chunks in chunks(requests, cfg.bs):
        batch_results = batch_edit_layerwise(
            cfg, model, tok, requests_chunks, device, cache_c, zs_full, z_layer, layers, last_layer
        )
        
        # Collect results for each layer
        for layer in layers:
            h_pre_current_all[layer].append(batch_results['h_pre_current'][layer])
            h_pre_last_at_layer[layer].append(batch_results['h_pre_last_at_layer'][layer])
            h_post_current_all[layer].append(batch_results['h_post_current'][layer])
            h_post_last_at_layer[layer].append(batch_results['h_post_last_at_layer'][layer])
    
    # Concatenate edited batches (h_pre_original already concatenated above)
    print(f"\nConcatenating hidden states from editing batches...")
    for layer in layers:
        h_pre_current_all[layer] = torch.cat(h_pre_current_all[layer], dim=1)
        h_pre_last_at_layer[layer] = torch.cat(h_pre_last_at_layer[layer], dim=1)
        h_post_current_all[layer] = torch.cat(h_post_current_all[layer], dim=1)
        h_post_last_at_layer[layer] = torch.cat(h_post_last_at_layer[layer], dim=1)
    
    # Save hidden states
    print(f"\n{'='*80}")
    print(f"SAVING HIDDEN STATES")
    print(f"{'='*80}")
    
    h_cache_dir = cfg.get('h_cache_dir', './h_cache')
    os.makedirs(h_cache_dir, exist_ok=True)
    
    # Save hidden states for each layer (before and after editing that layer)
    for layer in layers:
        # h_pre_original: layer i output from ORIGINAL model (completely unedited)
        print(f"Saving h_pre_original for layer {layer}, shape: {h_pre_original_all[layer].shape}")
        save_h_batch(
            h_cache_dir, dataset_name, model_cache_name, 
            'h_pre_original', h_pre_original_all[layer], 
            layer=layer, seed=seed_value, append=False
        )
        
        # h_pre_current: layer i output BEFORE editing layer i (with layers 0..i-1 already edited)
        print(f"Saving h_pre_current for layer {layer}, shape: {h_pre_current_all[layer].shape}")
        save_h_batch(
            h_cache_dir, dataset_name, model_cache_name, 
            'h_pre_current', h_pre_current_all[layer], 
            layer=layer, seed=seed_value, append=False
        )
        
        # h_pre_last_at_layer: last layer output BEFORE editing layer i
        print(f"Saving h_pre_last_at_layer{layer}, shape: {h_pre_last_at_layer[layer].shape}")
        save_h_batch(
            h_cache_dir, dataset_name, model_cache_name,
            f'h_pre_last_at_layer', h_pre_last_at_layer[layer],
            layer=layer, seed=seed_value, append=False
        )
        
        # h_post_current: layer i output AFTER editing layer i
        print(f"Saving h_post_current for layer {layer}, shape: {h_post_current_all[layer].shape}")
        save_h_batch(
            h_cache_dir, dataset_name, model_cache_name,
            'h_post_current', h_post_current_all[layer],
            layer=layer, seed=seed_value, append=False
        )
        
        # h_post_last_at_layer: last layer output AFTER editing layer i (cumulative effect)
        print(f"Saving h_post_last_at_layer{layer}, shape: {h_post_last_at_layer[layer].shape}")
        save_h_batch(
            h_cache_dir, dataset_name, model_cache_name,
            f'h_post_last_at_layer', h_post_last_at_layer[layer],
            layer=layer, seed=seed_value, append=False
        )
    
    # Print cache status
    print_h_cache_status(h_cache_dir, dataset_name, model_cache_name)
    
    return model


def batch_edit_layerwise(
    cfg, model, tok, requests, device, cache_c, zs_full, z_layer, layers, last_layer
) -> Dict:
    """
    Edit model weights layer-by-layer and extract hidden states after each edit.
    
    For each layer i being edited:
    - Extract h_pre_i and h_pre_last BEFORE editing layer i (with layers 0..i-1 already edited)
    - Apply edit to layer i
    - Extract h_post_i and h_post_last AFTER editing layer i (cumulative effect of layers 0..i)
    
    Returns:
        {
            'h_pre_current': {layer -> tensor [hidden_dim, batch_size]},
            'h_pre_last_at_layer': {layer -> tensor [hidden_dim, batch_size]},
            'h_post_current': {layer -> tensor [hidden_dim, batch_size]},
            'h_post_last_at_layer': {layer -> tensor [hidden_dim, batch_size]}
        }
    """
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    context_templates = get_context_templates(model, tok)
    
    # Extract z for current batch by sample indices
    sample_indices = [req["sample_idx"] for req in requests]
    max_cached_idx = zs_full.shape[1] - 1
    max_requested_idx = max(sample_indices)
    if max_requested_idx > max_cached_idx:
        raise IndexError(
            f"Sample index out of bounds! "
            f"Requested index: {max_requested_idx}, Max cached index: {max_cached_idx}. "
            f"Cache shape: {zs_full.shape}"
        )
    
    zs = zs_full[:, sample_indices].to(device)  # [hidden_dim, current_batch_size]
    print(f"Extracted z for batch indices {sample_indices[:5]}{'...' if len(sample_indices) > 5 else ''}, shape: {zs.shape}")
    
    # Initialize result storage
    result = {
        'h_pre_current': {},
        'h_pre_last_at_layer': {},
        'h_post_current': {},
        'h_post_last_at_layer': {}
    }
    
    # ========== Layer-by-layer editing with extraction ==========
    print(f"\n{'='*80}")
    print(f"LAYER-BY-LAYER EDITING WITH HIDDEN STATE EXTRACTION")
    print(f"{'='*80}")
    
    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n{'='*80}")
        print(f"PROCESSING LAYER {layer} ({i+1}/{len(cfg.llms.layers)})")
        print(f"{'='*80}")
        
        # ===== Extract h BEFORE editing current layer =====
        print(f"\n[BEFORE EDIT] Extracting h for layer {layer} and last layer {last_layer}")
        extraction_layers = [layer, last_layer] if layer != last_layer else [layer]
        h_before = extract_hidden_states(model, tok, requests, cfg, extraction_layers, device)
        
        result['h_pre_current'][layer] = h_before[layer]
        result['h_pre_last_at_layer'][layer] = h_before[last_layer]
        print(f"  ✓ h_pre_current[{layer}] shape: {result['h_pre_current'][layer].shape}")
        print(f"  ✓ h_pre_last_at_layer[{layer}] shape: {result['h_pre_last_at_layer'][layer].shape}")
        
        # ===== Apply MEMIT edit to current layer =====
        print(f"\n[EDITING] Applying MEMIT to layer {layer}")
        
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"  Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")
        
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
        
        targets = zs - cur_zs  # [dim, bs]
        print(f"  z error: {torch.linalg.norm(targets, dim=0).mean():.4f}")
        
        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        
        layer_ks, targets = (
            layer_ks.double(),
            targets.double()
        )
        resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        
        cov = covs[i].double()
        
        start_time = time.time()
        coef = cfg.llms.mom2_update_weight[i]
        upd_matrix = torch.linalg.solve(
            layer_ks @ layer_ks.T + cache_c[i, :, :].to(device).double() + coef * cov.to(device) +
            cfg.algs.L2 * torch.eye(layer_ks.shape[0], device=device).double(),
            layer_ks @ resid.T,
        )
        end_time = time.time()
        print(f"  Solved for update matrix in {end_time - start_time:.2f} seconds")
        
        if cfg.algs.add_old_keys:
            cache_c[i, :, :] += (layer_ks @ layer_ks.T).cpu()
        
        # Adjust update matrix shape and apply
        weight_name = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print(f"  Weight norm: orig={torch.linalg.norm(weights[weight_name]):.4f}, upd={torch.linalg.norm(upd_matrix):.4f}")
        
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        
        # ===== Extract h AFTER editing current layer =====
        print(f"\n[AFTER EDIT] Extracting h for layer {layer} and last layer {last_layer}")
        h_after = extract_hidden_states(model, tok, requests, cfg, extraction_layers, device)
        
        result['h_post_current'][layer] = h_after[layer]
        result['h_post_last_at_layer'][layer] = h_after[last_layer]
        print(f"  ✓ h_post_current[{layer}] shape: {result['h_post_current'][layer].shape}")
        print(f"  ✓ h_post_last_at_layer[{layer}] shape: {result['h_post_last_at_layer'][layer].shape}")
        
        # Calculate change
        h_change = torch.linalg.norm(result['h_post_current'][layer] - result['h_pre_current'][layer], dim=0).mean()
        last_change = torch.linalg.norm(result['h_post_last_at_layer'][layer] - result['h_pre_last_at_layer'][layer], dim=0).mean()
        print(f"  Δh_current[{layer}]: {h_change:.4f}")
        print(f"  Δh_last[{last_layer}] after editing layer {layer}: {last_change:.4f}")
    
    print(f"\n{'='*80}")
    print(f"COMPLETED ALL LAYERS")
    print(f"{'='*80}\n")
    
    return result


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
    """
    Get context templates for z computation.
    
    CRITICAL: This function MUST produce identical results in both:
    1. precompute_z.py (when caching z vectors)
    2. memit_main.py (when using cached z vectors)
    
    The templates are generated using generate_fast with a fixed seed.
    If the random seed or model state differs between precompute and inference,
    the z vectors will NOT match, causing large edit errors!
    """
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
