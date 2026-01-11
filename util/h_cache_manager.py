"""
H (hidden state) cache management module.
Handles storage and retrieval of layer outputs before/after editing.
Maintains format consistency with z_cache_manager.py.
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional
import time


def get_h_cache_filename(
    dataset: str,
    model_name: str,
    h_type: str,  # 'h_pre_current', 'h_pre_last', 'h_post_current', 'h_post_last'
    layer: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate standardized h cache filename.
    Format: {dataset}-{model_name}-{h_type}-layer{layer}-seed{seed}.pt
    
    h_type can be:
    - 'h_pre_original': 原始模型（完全未编辑）该层的输出
    - 'h_pre_current': 编辑前当前层的输出
    - 'h_pre_last': 编辑前最后编辑层的输出
    - 'h_pre_last_at_layer': 编辑到某层前，最后一层的输出
    - 'h_post_current': 编辑后当前层的输出
    - 'h_post_last': 编辑后最后编辑层的输出
    - 'h_post_last_at_layer': 编辑到某层后，最后一层的输出
    """
    model_name_clean = model_name.replace("/", "-")
    
    # Types that need layer number in filename
    types_with_layer = [
        'h_pre_original',
        'h_pre_current', 
        'h_post_current',
        'h_pre_last_at_layer',
        'h_post_last_at_layer'
    ]
    
    if h_type in types_with_layer:
        # These types need layer number
        if layer is None:
            raise ValueError(f"{h_type} requires layer number")
        if seed is not None:
            return f"{dataset}-{model_name_clean}-{h_type}-layer{layer}-seed{seed}.pt"
        else:
            return f"{dataset}-{model_name_clean}-{h_type}-layer{layer}.pt"
    else:
        # Last layer outputs don't need layer in filename (since it's always the last layer)
        if seed is not None:
            return f"{dataset}-{model_name_clean}-{h_type}-seed{seed}.pt"
        else:
            return f"{dataset}-{model_name_clean}-{h_type}.pt"


def get_h_cache_path(
    cache_dir: str,
    dataset: str,
    model_name: str,
    h_type: str,
    layer: Optional[int] = None,
    seed: Optional[int] = None,
) -> str:
    """Get full path to h cache file."""
    filename = get_h_cache_filename(dataset, model_name, h_type, layer, seed)
    return os.path.join(cache_dir, filename)


def get_h_metadata_path(cache_dir: str) -> str:
    """Get path to h metadata file that tracks cached h vectors."""
    return os.path.join(cache_dir, "h_metadata.json")


def load_h_metadata(cache_dir: str) -> Dict:
    """Load metadata about cached h vectors."""
    metadata_path = get_h_metadata_path(cache_dir)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def save_h_metadata(cache_dir: str, metadata: Dict) -> None:
    """Save metadata about cached h vectors."""
    os.makedirs(cache_dir, exist_ok=True)
    metadata_path = get_h_metadata_path(cache_dir)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def save_h_batch(
    cache_dir: str,
    dataset: str,
    model_name: str,
    h_type: str,
    h_vectors: torch.Tensor,
    layer: Optional[int] = None,
    seed: Optional[int] = None,
    append: bool = False,
) -> None:
    """
    Save h batch to cache.
    If append=True, appends to existing cache; otherwise overwrites.
    h_vectors shape should be [dim, batch_size] (same as z_cache_manager)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = get_h_cache_path(cache_dir, dataset, model_name, h_type, layer, seed)
    
    if append and os.path.exists(cache_path):
        existing_hs = torch.load(cache_path, map_location='cpu')
        # Concatenate along batch dimension
        h_vectors = torch.cat([existing_hs, h_vectors], dim=1)
    
    torch.save(h_vectors, cache_path)
    
    # Update metadata
    metadata = load_h_metadata(cache_dir)
    
    # Types that need layer number in metadata key
    types_with_layer = [
        'h_pre_original',
        'h_pre_current', 
        'h_post_current',
        'h_pre_last_at_layer',
        'h_post_last_at_layer'
    ]
    
    if h_type in types_with_layer:
        if seed is not None:
            key = f"{dataset}_{model_name}_{h_type}_layer{layer}_seed{seed}"
        else:
            key = f"{dataset}_{model_name}_{h_type}_layer{layer}"
    else:
        if seed is not None:
            key = f"{dataset}_{model_name}_{h_type}_seed{seed}"
        else:
            key = f"{dataset}_{model_name}_{h_type}"
    
    metadata[key] = {
        'dataset': dataset,
        'model_name': model_name,
        'h_type': h_type,
        'layer': layer,
        'seed': seed,
        'num_samples': h_vectors.shape[1],
        'dim': h_vectors.shape[0],
        'last_updated': str(time.time()),
    }
    save_h_metadata(cache_dir, metadata)


def load_h_batch(
    cache_dir: str,
    dataset: str,
    model_name: str,
    h_type: str,
    layer: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = 'cpu',
) -> Optional[torch.Tensor]:
    """
    Load h batch from cache.
    Returns shape [dim, num_samples] or None if not found.
    """
    cache_path = get_h_cache_path(cache_dir, dataset, model_name, h_type, layer, seed)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        h_vectors = torch.load(cache_path, map_location=device)
        return h_vectors
    except Exception as e:
        print(f"Error loading h cache from {cache_path}: {e}")
        return None


def get_cached_h_count(
    cache_dir: str,
    dataset: str,
    model_name: str,
    h_type: str,
    layer: Optional[int] = None,
    seed: Optional[int] = None,
) -> int:
    """Get count of cached h vectors for given configuration."""
    cache_path = get_h_cache_path(cache_dir, dataset, model_name, h_type, layer, seed)
    
    if not os.path.exists(cache_path):
        return 0
    
    try:
        hs = torch.load(cache_path, map_location='cpu')
        # hs shape should be [dim, num_samples]
        if len(hs.shape) == 2:
            return hs.shape[1]
        return 0
    except Exception as e:
        print(f"Error loading cached h file {cache_path}: {e}")
        return 0


def print_h_cache_status(cache_dir: str, dataset: str, model_name: str) -> None:
    """Print status of h cache."""
    metadata = load_h_metadata(cache_dir)
    
    print(f"\n{'='*80}")
    print(f"H CACHE STATUS")
    print(f"{'='*80}")
    print(f"Cache directory: {cache_dir}")
    print(f"Dataset: {dataset}")
    print(f"Model: {model_name}")
    print(f"\nCached entries:")
    
    if not metadata:
        print("  (empty)")
        return
    
    # Group by h_type
    h_types = {}
    for key, info in metadata.items():
        if info['dataset'] == dataset and info['model_name'] == model_name:
            h_type = info['h_type']
            if h_type not in h_types:
                h_types[h_type] = []
            h_types[h_type].append(info)
    
    for h_type in sorted(h_types.keys()):
        entries = h_types[h_type]
        print(f"\n  {h_type}:")
        for entry in entries:
            if entry.get('layer') is not None:
                print(f"    Layer {entry['layer']}: {entry['num_samples']} samples, dim={entry['dim']}, seed={entry.get('seed', 'N/A')}")
            else:
                print(f"    {entry['num_samples']} samples, dim={entry['dim']}, seed={entry.get('seed', 'N/A')}")
    
    print("="*80)
