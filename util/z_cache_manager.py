"""
Unified Z computation and cache management module.
Handles pre-computation, caching, and retrieval of z vectors.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
import time


def get_z_cache_filename(
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> str:
    """
    Generate standardized z cache filename.
    Format: {dataset}-{model_name}-{z_method}-seed{seed}-layer{layer}.pt
    If seed is None, omits the seed part (for backward compatibility).
    """
    model_name_clean = model_name.replace("/", "-")
    # if seed is not None:
    #     return f"{dataset}-{model_name_clean}-{z_method}-seed{seed}-layer{layer}.pt"
    # else:
    #     return f"{dataset}-{model_name_clean}-{z_method}-layer{layer}.pt"

    filename = f"{dataset}-{model_name_clean}-{z_method}"
    if seed is not None:
        filename += f"-seed{seed}"
    if v_lr is not None:
        filename += f"-vlr{v_lr}"
    if steps is not None:
        filename += f"-steps{steps}"
    filename += f"-layer{layer}.pt"
    return filename
    


def get_z_cache_path(
    cache_dir: str,
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> str:
    """Get full path to z cache file."""
    filename = get_z_cache_filename(dataset, model_name, z_method, layer, seed,v_lr,steps)
    return os.path.join(cache_dir, filename)


def get_z_metadata_path(cache_dir: str) -> str:
    """Get path to z metadata file that tracks cached z vectors."""
    return os.path.join(cache_dir, "z_metadata.json")


def load_z_metadata(cache_dir: str) -> Dict:
    """Load metadata about cached z vectors."""
    metadata_path = get_z_metadata_path(cache_dir)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def save_z_metadata(cache_dir: str, metadata: Dict) -> None:
    """Save metadata about cached z vectors."""
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = get_z_metadata_path(cache_dir)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def get_cached_z_count(
    cache_dir: str,
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> int:
    """Get count of cached z vectors for given configuration."""
    cache_path = get_z_cache_path(cache_dir, dataset, model_name, z_method, layer, seed, v_lr, steps)
    if not os.path.exists(cache_path):
        return 0
    
    try:
        zs = torch.load(cache_path, map_location='cpu')
        # zs shape should be [dim, num_samples]
        if len(zs.shape) == 2:
            return zs.shape[1]
        return 0
    except Exception as e:
        print(f"Error loading cached z file {cache_path}: {e}")
        return 0


def save_z_batch(
    cache_dir: str,
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    z_vectors: torch.Tensor,
    append: bool = True,
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> None:
    """
    Save z batch to cache.
    If append=True, appends to existing cache; otherwise overwrites.
    z_vectors shape should be [dim, batch_size]
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_z_cache_path(cache_dir, dataset, model_name, z_method, layer, seed, v_lr, steps)
    
    if append and os.path.exists(cache_path):
        existing_zs = torch.load(cache_path, map_location='cpu')
        # Concatenate along batch dimension
        z_vectors = torch.cat([existing_zs, z_vectors], dim=1)
    
    torch.save(z_vectors, cache_path)
    
    # Update metadata
    metadata = load_z_metadata(cache_dir)
    if seed is not None:
        key = f"{dataset}_{model_name}_{z_method}_seed{seed}_layer{layer}"
    else:
        key = f"{dataset}_{model_name}_{z_method}_layer{layer}"
    metadata[key] = {
        'dataset': dataset,
        'model_name': model_name,
        'z_method': z_method,
        'layer': layer,
        'seed': seed,
        'num_samples': z_vectors.shape[1],
        'dim': z_vectors.shape[0],
        'last_updated': str(time.time()),
        'v_lr': v_lr,
        'steps': steps,
    }
    save_z_metadata(cache_dir, metadata)


def load_z_batch(
    cache_dir: str,
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    device: str = 'cpu',
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    Load z batch from cache.
    Returns shape [dim, num_samples] or None if not found.
    """
    cache_path = get_z_cache_path(cache_dir, dataset, model_name, z_method, layer, seed, v_lr, steps)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        zs = torch.load(cache_path, map_location=device)
        return zs
    except Exception as e:
        print(f"Error loading z from {cache_path}: {e}")
        return None


def get_z_list_and_count(
    cache_dir: str,
    dataset: str,
    model_name: str,
    z_method: str,
    layer: int,
    target_count: int,
    seed: Optional[int] = None,
    v_lr: Optional[float] = None,
    steps: Optional[int] = None,
) -> Tuple[torch.Tensor, int, int]:
    """
    Get cached z vectors and return (z_tensor, cached_count, needed_count).
    
    Returns:
        z_tensor: Cached z vectors [dim, cached_count] or None
        cached_count: Number of already cached z vectors
        needed_count: Number of additional z vectors needed to reach target_count
    """
    cached_count = get_cached_z_count(cache_dir, dataset, model_name, z_method, layer, seed, v_lr, steps)
    needed_count = max(0, target_count - cached_count)
    
    z_tensor = None
    if cached_count > 0:
        z_tensor = load_z_batch(cache_dir, dataset, model_name, z_method, layer, seed=seed, v_lr=v_lr, steps=steps)
    
    return z_tensor, cached_count, needed_count


def list_available_z_methods(cache_dir: str, dataset: str, model_name: str, layer: int) -> List[str]:
    """List all available z methods for given dataset, model, and layer."""
    metadata = load_z_metadata(cache_dir)
    model_name_clean = model_name.replace("/", "-")
    
    methods = []
    for key in metadata.keys():
        if dataset in key and model_name_clean in key and f"layer{layer}" in key:
            # Extract z_method from key: {dataset}_{model_name}_{z_method}_layer{layer}
            parts = key.split('_layer')[0].split('_')
            # Remove dataset and model_name parts
            method = '_'.join(parts[len(dataset.split('_')) + len(model_name_clean.split('-')):])
            if method:
                methods.append(method)
    
    return list(set(methods))


def get_z_info(cache_dir: str, dataset: str, model_name: str, z_method: str, layer: int) -> Optional[Dict]:
    """Get metadata info for specific z configuration."""
    metadata = load_z_metadata(cache_dir)
    model_name_clean = model_name.replace("/", "-")
    key = f"{dataset}_{model_name_clean}_{z_method}_layer{layer}"
    return metadata.get(key)


def print_z_cache_status(cache_dir: str, dataset: Optional[str] = None, 
                         model_name: Optional[str] = None, layer: Optional[int] = None, seed: Optional[int] = None, v_lr: Optional[float] = None, steps: Optional[int] = None) -> None:
    """Print status of z cache."""
    metadata = load_z_metadata(cache_dir)
    
    if not metadata:
        print("No z cache found.")
        return
    
    print("\n" + "="*80)
    print("Z CACHE STATUS")
    print("="*80)
    
    for key, info in metadata.items():
        if dataset and dataset not in key:
            continue
        if model_name and model_name.replace("/", "-") not in key:
            continue
        if layer is not None and f"layer{layer}" not in key:
            continue
        print(f"\nDataset: {info['dataset']}")
        print(f"Model: {info['model_name']}")
        print(f"Z Method: {info['z_method']}")
        print(f"Layer: {info['layer']}")
        print(f"Num Samples: {info['num_samples']}")
        print(f"Dimension: {info['dim']}")
        print(f"Last Updated: {info['last_updated']}")
        print(f"Seed: {info['seed']}")
        print(f"v_lr: {info.get('v_lr', 'N/A')}")
        print(f"v_num_grad_steps: {info.get('v_num_grad_steps', 'N/A')}")
    
    print("\n" + "="*80)
