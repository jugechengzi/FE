"""
Pre-compute and cache z vectors using specified methods.
Supports different computation strategies (all_layers, first_forward).
"""

import os
import torch
import json
import argparse
import sys
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import random

from util.z_cache_manager import (
    save_z_batch, get_cached_z_count, get_z_cache_path, 
    print_z_cache_status
)
from load import load_data
from z_methods import get_z_compute_function, get_z_strategy, find_fact_lookup_idx
from util import nethook
from util.generate import generate_fast

# Global variable to store edited_layers parsed before Hydra init
_edited_layers_global = None
# Cache variable for context templates
CONTEXT_TEMPLATES_CACHE = None


def parse_edited_layers_from_argv():
    """
    Parse edited_layers from command line arguments.
    Format: --edited_layers=4,5,6,7,8
    This parameter is REQUIRED and must be extracted BEFORE Hydra initialization.
    Removes it from sys.argv to avoid Hydra parsing conflicts.
    """
    edited_layers_arg = None
    
    # Search for edited_layers in sys.argv
    for i, arg in enumerate(sys.argv):
        if arg.startswith('--edited_layers='):
            edited_layers_arg = arg
            # Remove from sys.argv so Hydra doesn't see it
            sys.argv.pop(i)
            break
    
    if edited_layers_arg is None:
        print("Error: --edited_layers is REQUIRED!")
        print("Usage: python precompute_z.py --edited_layers=0,15,31 z_method=all num_z_samples=2000")
        sys.exit(1)
    
    # Extract the value part
    try:
        value_str = edited_layers_arg.split('=', 1)[1]
        edited_layers = [int(x.strip()) for x in value_str.split(',')]
        edited_layers = sorted(edited_layers)
        return edited_layers
    except (ValueError, IndexError) as e:
        print(f"Error parsing edited_layers: {e}")
        print(f"Expected format: --edited_layers=0,15,31")
        sys.exit(1)


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_context_templates(model, tok):
    """
    Get context templates based on model configuration.
    IMPORTANT: This MUST match the implementation in memit_main.py!
    Uses generate_fast to create templates, ensuring consistency with editing phase.
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


def compute_z_batch_final_layer(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: List[Dict],
    cfg: DictConfig,
    z_method: str,
    z_layer: int,
    num_samples: int,
    context_templates: List,
) -> torch.Tensor:
    """
    Compute z for final layer only (strategy: final_layer).
    Returns z tensor of shape [hidden_dim, num_samples].
    """
    compute_z = get_z_compute_function(z_method)
    
    z_list = []
    for i, request in enumerate(tqdm(data[:num_samples], desc=f"Computing z at layer {z_layer}")):
        if i >= num_samples:
            break
        
        # CRITICAL: Add space prefix to target_new (same as memit_main.py)
        request_copy = dict(request)
        request_copy["target_new"] = " " + request["target_new"]
        
        try:
            cur_z = compute_z(
                model,
                tok,
                request_copy,
                cfg,
                z_layer,
                context_templates,
            )
            z_list.append(cur_z)
        except Exception as e:
            print(f"Error computing z for sample {i}: {e}")
            continue
    
    if not z_list:
        raise RuntimeError("Failed to compute any z vectors!")
    
    zs = torch.stack(z_list, dim=1)  # [hidden_dim, num_samples]
    return zs


def compute_z_batch_all_layers(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: List[Dict],
    cfg: DictConfig,
    z_method: str,
    layers: List[int],
    num_samples: int,
    context_templates: List,
) -> Dict[int, torch.Tensor]:
    """
    Compute z for all edit layers (strategy: all_layers).
    Returns dict mapping layer -> z tensor of shape [hidden_dim, num_samples].
    """
    compute_z = get_z_compute_function(z_method)
    
    zs_dict = {layer: [] for layer in layers}
    
    for sample_idx, request in enumerate(tqdm(data[:num_samples], desc="Computing z at all layers")):
        if sample_idx >= num_samples:
            break
        
        # CRITICAL: Add space prefix to target_new (same as memit_main.py)
        request_copy = dict(request)
        request_copy["target_new"] = " " + request["target_new"]
        
        for layer in layers:
            try:
                cur_z = compute_z(
                    model,
                    tok,
                    request_copy,
                    cfg,
                    layer,
                    context_templates,
                )
                zs_dict[layer].append(cur_z)
            except Exception as e:
                print(f"Error computing z for sample {sample_idx} at layer {layer}: {e}")
                # Append None as placeholder to maintain indexing
                zs_dict[layer].append(None)
    
    # Stack z vectors for each layer
    result = {}
    for layer in layers:
        # Filter out None values but maintain order
        valid_zs = [z for z in zs_dict[layer] if z is not None]
        if not valid_zs:
            raise RuntimeError(f"Failed to compute any z vectors for layer {layer}!")
        result[layer] = torch.stack(valid_zs, dim=1)  # [hidden_dim, num_valid_samples]
    
    return result


def compute_z_batch_first_forward(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: List[Dict],
    cfg: DictConfig,
    z_method: str,
    layers: List[int],
    num_samples: int,
    context_templates: List,
) -> Dict[int, torch.Tensor]:
    """
    Compute z at first layer, then propagate forward to get z at other layers (strategy: first_forward).
    Returns dict mapping layer -> z tensor of shape [hidden_dim, num_samples].
    
    This is inspired by memit_a_main.py's forward propagation strategy.
    Automatically detects first layer as min(layers).
    """
    compute_z = get_z_compute_function(z_method)
    
    # layers already sorted by caller, but ensure it
    layers_sorted = sorted(layers)
    first_layer = layers_sorted[0]
    other_layers = layers_sorted[1:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Step 1: Compute z at first layer
    print(f"\n{'='*80}")
    print(f"STEP 1: Computing z at first layer {first_layer}")
    print(f"{'='*80}")
    
    z_first_layer_list = []
    for sample_idx, request in enumerate(tqdm(data[:num_samples], desc=f"Computing z at first layer {first_layer}")):
        if sample_idx >= num_samples:
            break
        
        # CRITICAL: Add space prefix to target_new (same as memit_main.py)
        request_copy = dict(request)
        request_copy["target_new"] = " " + request["target_new"]
        
        try:
            cur_z = compute_z(
                model,
                tok,
                request_copy,
                cfg,
                first_layer,
                context_templates,
            )
            z_first_layer_list.append((sample_idx, cur_z))
        except Exception as e:
            print(f"Error computing z for sample {sample_idx}: {e}")
            continue
    
    if not z_first_layer_list:
        raise RuntimeError("Failed to compute any z vectors!")
    
    z_first = torch.stack([z for _, z in z_first_layer_list], dim=1)  # [hidden_dim, num_samples]
    
    # Step 2: Forward propagate to get z at other layers
    print(f"\n{'='*80}")
    print(f"STEP 2: Forward propagating to layers {other_layers}")
    print(f"{'='*80}")
    
    zs_dict = {first_layer: z_first}
    
    for sample_idx, request in enumerate(tqdm(data[:num_samples], desc="Forward propagating z")):

        # # Forward propagation through layers
        # prompt = [
        #     context.format(request["prompt"])
        #     for context_types in context_templates
        #     for context in context_types
        # ]
        # input_tok = tok(
        #     [p.format(request["subject"]) for p in prompt],
        #     return_tensors="pt",
        #     padding=True,
        # ).to(device)

        # # Find lookup index
        # fact_token = cfg.llms.fact_token
        # lookup_idx = [
        #     find_fact_lookup_idx(
        #         prompt, request["subject"], tok, fact_token, verbose=False
        #     )
        #     for _, prompt in enumerate(prompt)
        # ]

        # def edit_output_fn(cur_out, cur_layer_name):
        #     target_hidden = cur_out[0] if isinstance(cur_out, tuple) else cur_out

        #     first_layer_module = cfg.llms.layer_module_tmp.format(first_layer)
        #     # Only inject z at the first layer
        #     # For all prompts, inject the same zs from first layer computation
        #     if cur_layer_name == first_layer_module:
        #         z_to_inject = zs_dict[first_layer][:, sample_idx]
        #         if sample_idx == 0: # 只打印第一个样本
        #             print(f"\n[DEBUG Layer {cur_layer_name}]")
        #             print(f"  target_hidden shape: {target_hidden.shape}")
        #             print(f"  z_to_inject shape: {z_to_inject.shape}")
        #             print(f"  lookup_idx: {lookup_idx}")

        #         for batch_idx, seq_idx in enumerate(lookup_idx):

        #             if target_hidden.shape[0]==len(lookup_idx):
        #                 target_hidden[batch_idx, seq_idx, :] = z_to_inject
        #             else:
        #                 target_hidden[seq_idx, batch_idx, :] = z_to_inject
        #     return cur_out

        # with torch.no_grad():
        #     with nethook.TraceDict(
        #         module=model,
        #         layers=[cfg.llms.layer_module_tmp.format(l) for l in layers_sorted],  # Must trace ALL layers including first_layer!
        #         edit_output=edit_output_fn,
        #     ) as tr:
        #         model(**input_tok)

        #     for layer in other_layers:
        #         layer_module = cfg.llms.layer_module_tmp.format(layer)
        #         output = tr[layer_module].output
        #         c_z_raw = output[0] if isinstance(output, tuple) else output

        #         # Extract at lookup positions (match memit_a_main.py logic)
        #         c_z_list = []
        #         for batch_idx in range(len(lookup_idx)):
        #             cur_lookup_idx = lookup_idx[batch_idx]
        #             if c_z_raw.dim() == 3:
        #                 if c_z_raw.shape[0] == len(lookup_idx):
        #                     c_z = c_z_raw[batch_idx, cur_lookup_idx, :]
        #                 else:
        #                     c_z = c_z_raw[cur_lookup_idx, batch_idx, :]
        #             else:
        #                 c_z = c_z_raw[cur_lookup_idx, :]
        #             c_z_list.append(c_z.detach().clone())
                    
        #         # Average over prompts to get single z per sample
        #         c_z_avg = torch.stack(c_z_list, dim=0).mean(dim=0)

        #         if layer not in zs_dict:
        #             zs_dict[layer] = []
        #         zs_dict[layer].append(c_z_avg)

        # Forward propagate through layers
        prompt = request["prompt"].format(request["subject"])
        input_tok = tok(prompt, return_tensors="pt").to(device)
        
        # Find lookup index
        fact_token = cfg.llms.fact_token
        lookup_idx = find_fact_lookup_idx(request["prompt"], request["subject"], tok, fact_token, verbose=False)
        
        def edit_output_fn(cur_out, cur_layer_name):
            first_layer_module = cfg.llms.layer_module_tmp.format(first_layer)
            # 仅在第一层需要修改的层处注入z
            if cur_layer_name == first_layer_module:
                z_to_inject = zs_dict[first_layer][:, sample_idx]
                if cur_out[0].shape[0] == 1:
                    cur_out[0][0, lookup_idx, :] = z_to_inject
                else:
                    cur_out[0][lookup_idx, 0, :] = z_to_inject
            return cur_out
        
        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=[cfg.llms.layer_module_tmp.format(l) for l in layers_sorted],  # Must trace ALL layers including first_layer!
                edit_output=edit_output_fn,
            ) as tr:
                model(**input_tok)
            
            # Extract z from each layer
            for layer in other_layers:
                layer_module = cfg.llms.layer_module_tmp.format(layer)
                output = tr[layer_module].output
                c_z_raw = output[0] if isinstance(output, tuple) else output
                
                # Extract at lookup position (match memit_a_main.py logic)
                if c_z_raw.dim() == 3:
                    c_z = c_z_raw[0, lookup_idx, :] if c_z_raw.shape[0] == 1 else c_z_raw[lookup_idx, 0, :]
                else:
                    c_z = c_z_raw[lookup_idx, :]
                
                if layer not in zs_dict:
                    zs_dict[layer] = []
                zs_dict[layer].append(c_z.detach().clone())
    
    # Stack propagated z vectors
    for layer in other_layers:
        if layer in zs_dict and isinstance(zs_dict[layer], list):
            zs_dict[layer] = torch.stack(zs_dict[layer], dim=1)
    
    return zs_dict


@hydra.main(config_path="configs", config_name="config", version_base=None)
def precompute_z(cfg: DictConfig) -> None:
    """
    Pre-compute z vectors for specified configuration.
    
    IMPORTANT: --edited_layers is REQUIRED via command line!
    
    Usage:
        # Compute z for all specified edit layers
        python precompute_z.py --edited_layers=0,15,31 z_method=all num_z_samples=2000
        
        # Compute z using first-forward strategy
        python precompute_z.py --edited_layers=0,15,31 z_method=firstforward num_z_samples=2000
    """
    # Parse edited_layers from command line (REQUIRED)
    # This was already extracted before Hydra init, just retrieve it from globals
    edited_layers = _edited_layers_global
    
    print(OmegaConf.to_yaml(cfg))
    print(f"\n{'='*80}")
    print(f"Edited layers from command line: {edited_layers}")
    print(f"{'='*80}\n")
    
    # Get parameters
    z_method = cfg.get('z_method', 'all')
    num_z_samples = cfg.get('num_z_samples', cfg.num_edits)
    dataset = cfg.data
    model_name = cfg.llms.name
    
    # Get z method strategy
    try:
        strategy = get_z_strategy(z_method)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print(f"\n{'='*80}")
    print(f"PRE-COMPUTING Z VECTORS")
    print(f"{'='*80}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"Z Method: {z_method}")
    print(f"Strategy: {strategy}")
    print(f"Edit Layers: {edited_layers}")
    print(f"Target Samples: {num_z_samples}")
    print(f"Cache Dir: {cfg.zs_cache_dir}")

    # Load v_lr and steps if applicable
    v_lr = cfg.llms.get('v_lr', None)
    steps = cfg.llms.get('v_num_grad_steps', None)
    print(f"v_lr: {v_lr}, v_num_grad_steps: {steps}")
    
    # Set random seed 这里的seed是全局的seed，在这里设置好之后，后续所有的计算都会使用这个seed，包括z的计算
    set_random_seed(cfg.seed)
    print(f"\nRandom seed set to {cfg.seed}")
    
    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading model to {device}...")
    
    if cfg.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.model_dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.model_dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    # IMPORTANT: z precomputation must run with dropout disabled.
    # This should match edit-time behavior (model.eval()) to avoid RNG-driven drift.
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Setup tokenizer
    if "QWenTokenizer" in str(type(tok)):
        target_token = "<|endoftext|>"
        tok.eos_token = target_token
        tok.pad_token = target_token
        tok.bos_token = target_token
    
    if tok.pad_token is None:
        try:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        except AttributeError:
            tok.pad_token_id = tok.eos_token_id
    
    print("Model loaded successfully!")
    
    # CRITICAL: Initialize context templates AFTER model is loaded and seed is set
    # This ensures generate_fast produces the same templates as in memit_main.py
    print("\nInitializing context templates (this affects z computation)...")
    context_templates = get_context_templates(model, tok)
    print(f"Context templates initialized: {len(context_templates)} template groups")
    
    # Load data
    print(f"\nLoading dataset {dataset}...")
    data = load_data(cfg)
    print(f"Loaded {len(data)} samples")
    
    # Execute computation based on strategy
    if strategy == "all_layers":
        # Check cache for all layers
        all_cached = True
        for layer in edited_layers:
            cached_count = get_cached_z_count(cfg.zs_cache_dir, dataset, model_name, z_method, layer, cfg.seed, v_lr, steps)
            if cached_count < num_z_samples:
                all_cached = False
                break
        
        if all_cached:
            print(f"Cache already contains all required samples for all layers")
            print_z_cache_status(cfg.zs_cache_dir, dataset, model_name)
            return
        
        # Compute z for all layers
        zs_dict = compute_z_batch_all_layers(model, tok, data, cfg, z_method, edited_layers, num_z_samples, context_templates)
        
        for layer, zs in zs_dict.items():
            print(f"\nSaving z for layer {layer}, shape: {zs.shape}")
            save_z_batch(cfg.zs_cache_dir, dataset, model_name, z_method, layer, zs, append=False, seed=cfg.seed, v_lr=v_lr, steps=steps)
        
        print_z_cache_status(cfg.zs_cache_dir, dataset, model_name, v_lr, steps)
    
    elif strategy == "first_forward":
        # Auto-detect first layer
        first_layer = min(edited_layers)
        other_layers = [l for l in edited_layers if l != first_layer]
        
        # Check cache for first layer and others
        first_cached = get_cached_z_count(cfg.zs_cache_dir, dataset, model_name, z_method, first_layer, cfg.seed, v_lr, steps)
        
        if first_cached >= num_z_samples:
            all_other_cached = True
            for layer in other_layers:
                other_cached_count = get_cached_z_count(cfg.zs_cache_dir, dataset, model_name, z_method, layer, cfg.seed, v_lr, steps)
                if other_cached_count < num_z_samples:
                    all_other_cached = False
                    break
            
            if all_other_cached:
                print(f"Cache already contains all required samples for all layers")
                print_z_cache_status(cfg.zs_cache_dir, dataset, model_name, v_lr, steps)
                return
        
        print(f"First layer auto-detected: {first_layer}")
        print(f"Other layers: {other_layers}")
        
        # Compute using forward propagation strategy
        zs_dict = compute_z_batch_first_forward(model, tok, data, cfg, z_method, edited_layers, num_z_samples, context_templates)
        
        for layer, zs in zs_dict.items():
            print(f"\nSaving z for layer {layer}, shape: {zs.shape}")
            save_z_batch(cfg.zs_cache_dir, dataset, model_name, z_method, layer, zs, append=False, seed=cfg.seed, v_lr=v_lr, steps=steps)
        
        print_z_cache_status(cfg.zs_cache_dir, dataset, model_name, v_lr, steps)
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print("\n" + "="*80)
    print("Z pre-computation completed!")
    print("="*80)


if __name__ == "__main__":
    # Parse edited_layers BEFORE Hydra initialization
    _edited_layers_global = parse_edited_layers_from_argv()
    
    precompute_z()
