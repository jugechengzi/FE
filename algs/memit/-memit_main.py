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

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

def apply_memit_to_model(
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
        # Add sample index if not already present (for z extraction)
        if "sample_idx" not in request:
            requests[i]["sample_idx"] = i
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    #查看KKT是否已经计算好。
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +("-" if cfg.cache_filename_suffix !="" else "") + cfg.cache_filename_suffix + ".npz"
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
    
    # New approach: Load layer-specific z caches dynamically in batch_edit
    # Prepare z cache metadata for validation
    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = getattr(cfg, 'data', 'unknown')
    seed_value = getattr(cfg, 'seed', 0)
    z_method = "firstforward"  # Default to firstforward method
    
    # Validate that all required layer-specific z caches exist
    print(f"\nValidating layer-specific z caches (method: {z_method})...")
    for layer in cfg.llms.layers:
        cache_zs_file = f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{layer}.pt"
        if not os.path.isfile(cache_zs_file):
            raise FileNotFoundError(
                f"Layer-specific z cache not found: {cache_zs_file}\n"
                f"Please pre-compute z for layer {layer} using precompute_z.py with method={z_method}"
            )
        # Quick validation of sample count
        zs_temp = torch.load(cache_zs_file, map_location='cpu')
        num_cached_samples = zs_temp.shape[1]
        num_required_samples = len(requests)
        if num_cached_samples < num_required_samples:
            raise ValueError(
                f"Insufficient cached z samples for layer {layer}! "
                f"Required: {num_required_samples}, Cached: {num_cached_samples}"
            )
        print(f"  ✓ Layer {layer}: {cache_zs_file} ({num_cached_samples} samples)")
    print("✓ All layer-specific z caches validated\n")
    
    # Prepare z cache config for batch_edit
    z_cache_config = {
        'dataset_name': dataset_name,
        'model_cache_name': model_cache_name,
        'z_method': z_method,
        'seed_value': seed_value,
        'cache_dir': cfg.zs_cache_dir
    }
    
    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg, model, tok, requests_chunks, device, cache_c, z_cache_config)
    return model

def batch_edit(cfg, model, tok, requests, device, cache_c, z_cache_config):
    """
    Edit model weights for a batch of requests.
    
    Args:
        z_cache_config: Dictionary containing cache metadata:
            - dataset_name: Name of the dataset
            - model_cache_name: Model name (sanitized)
            - z_method: Z computation method (e.g., 'firstforward')
            - seed_value: Random seed used
            - cache_dir: Base directory for z caches
    
    New behavior: Load layer-specific z caches dynamically for each layer.
    """
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    context_templates = get_context_templates(model, tok)
    
    # Extract sample indices for current batch
    sample_indices = [req["sample_idx"] for req in requests]
    print(f"Processing batch with {len(sample_indices)} samples: {sample_indices[:5]}{'...' if len(sample_indices) > 5 else ''}")


    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\n{'='*60}")
        print(f"LAYER {layer} (Index {i}/{len(cfg.llms.layers)})")
        print(f"{'='*60}")
        
        # Load layer-specific z cache
        cache_zs_file = (
            f"{z_cache_config['cache_dir']}/{z_cache_config['dataset_name']}-"
            f"{z_cache_config['model_cache_name']}-{z_cache_config['z_method']}-"
            f"seed{z_cache_config['seed_value']}-layer{layer}.pt"
        )
        print(f"Loading layer-specific z cache: {cache_zs_file}")
        zs_layer_full = torch.load(cache_zs_file, map_location='cpu')  # [hidden_dim, num_all_samples]
        
        # Extract z for current batch by sample indices
        zs = zs_layer_full[:, sample_indices].to(device)  # [hidden_dim, current_batch_size]
        print(f"  ✓ Loaded z for layer {layer}, shape: {zs.shape}")
        
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute current layer output (cur_zs)
        if cfg.negetive_prompt_test:
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                layer,  # Use current layer instead of fixed z_layer
                context_templates=[request["negetive_prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        else:
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                layer,  # Use current layer instead of fixed z_layer
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        
        # Compute targets: difference between pre-computed z and current output
        targets = zs - cur_zs  # [dim, bs]
        avg_target_norm = torch.linalg.norm(targets, dim=0).mean().item()
        avg_z_norm = torch.linalg.norm(zs, dim=0).mean().item()
        avg_curz_norm = torch.linalg.norm(cur_zs, dim=0).mean().item()
        print(f"  Target stats: ||target||={avg_target_norm:.4f}, ||z||={avg_z_norm:.4f}, ||cur_z||={avg_curz_norm:.4f}")

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        layer_ks, targets = (
            layer_ks.double(),
            targets.double()
        )

        # resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        resid = targets  # No distribution

        cov = covs[i].double()

        start_time = time.time()
        coef=cfg.llms.mom2_update_weight[i]
        upd_matrix = torch.linalg.solve(
            layer_ks @ layer_ks.T + cache_c[i, :, :].to(device).double()+coef*cov.to(device)+
            cfg.algs.L2 * torch.eye(layer_ks.shape[0], device=device).double(),
            layer_ks @ resid.T,
        )
        end_time = time.time()
        print(f"Solved for update matrix in {end_time - start_time:.2f} seconds")
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
