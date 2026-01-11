"""
Compare hidden states (h) with target z vectors.

Implements two key experiments:
1. Update ratio: ||h_post_i - z_i|| / ||h_pre_i - z_i||
   - Measures how much closer we get to ideal state after editing
   
2. Cosine similarity: cos(z_last - h_pre_i_last, h_post_i_last - h_pre_i_last)
   - Measures alignment between edit direction and ideal direction

Results are saved to JSON with detailed per-sample information for visualization.
"""

import os
import torch
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import datetime


def load_h_cache(h_cache_dir: str, dataset: str, model_name: str, h_type: str, layer: int, seed: int) -> torch.Tensor:
    """Load h cache file."""
    model_name_clean = model_name.replace("/", "-")
    filename = f"{dataset}-{model_name_clean}-{h_type}-layer{layer}-seed{seed}.pt"
    filepath = os.path.join(h_cache_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"H cache not found: {filepath}")
    
    return torch.load(filepath, map_location='cpu')


def load_z_cache(z_cache_dir: str, dataset: str, model_name: str, z_method: str, layer: int, seed: int) -> torch.Tensor:
    """Load z cache file."""
    model_name_clean = model_name.replace("/", "-")
    filename = f"{dataset}-{model_name_clean}-{z_method}-seed{seed}-layer{layer}.pt"
    filepath = os.path.join(z_cache_dir, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Z cache not found: {filepath}")
    
    return torch.load(filepath, map_location='cpu')


def compute_update_ratio(
    h_pre: torch.Tensor,
    h_post: torch.Tensor,
    z: torch.Tensor
) -> Tuple[np.ndarray, Dict]:
    """
    Compute update ratio: ||h_post - z|| / ||h_pre - z||
    
    Args:
        h_pre: [hidden_dim, num_samples]
        h_post: [hidden_dim, num_samples]
        z: [hidden_dim, num_samples]
    
    Returns:
        ratios: array of shape [num_samples]
        stats: dict with statistics
    """
    # Compute distances for each sample
    dist_pre = torch.linalg.norm(h_pre - z, dim=0)  # [num_samples]
    dist_post = torch.linalg.norm(h_post - z, dim=0)  # [num_samples]
    
    # Avoid division by zero
    ratios = (dist_post / (dist_pre + 1e-10)).detach().float().numpy()
    
    stats = {
        'mean': float(ratios.mean()),
        'std': float(ratios.std()),
        'min': float(ratios.min()),
        'max': float(ratios.max()),
        'median': float(np.median(ratios)),
        'q25': float(np.percentile(ratios, 25)),
        'q75': float(np.percentile(ratios, 75)),
        'num_improved': int((ratios < 1.0).sum()),  # ratio < 1 means closer to z
        'num_degraded': int((ratios > 1.0).sum()),  # ratio > 1 means farther from z
        'improvement_rate': float((ratios < 1.0).mean() * 100),  # percentage
    }
    
    return ratios, stats


def compute_cosine_similarity(
    ideal_direction: torch.Tensor,
    edit_direction: torch.Tensor
) -> Tuple[np.ndarray, Dict]:
    """
    Compute cosine similarity between ideal direction and edit direction.
    
    Args:
        ideal_direction: [hidden_dim, num_samples], z_last - h_pre_i_last
        edit_direction: [hidden_dim, num_samples], h_post_i_last - h_pre_i_last
    
    Returns:
        similarities: array of shape [num_samples]
        stats: dict with statistics
    """
    # Compute cosine similarity for each sample
    dot_products = (ideal_direction * edit_direction).sum(dim=0)  # [num_samples]
    norm_ideal = torch.linalg.norm(ideal_direction, dim=0)  # [num_samples]
    norm_edit = torch.linalg.norm(edit_direction, dim=0)  # [num_samples]
    
    # Avoid division by zero
    similarities = (dot_products / (norm_ideal * norm_edit + 1e-10)).detach().float().numpy()
    
    stats = {
        'mean': float(similarities.mean()),
        'std': float(similarities.std()),
        'min': float(similarities.min()),
        'max': float(similarities.max()),
        'median': float(np.median(similarities)),
        'q25': float(np.percentile(similarities, 25)),
        'q75': float(np.percentile(similarities, 75)),
        'num_positive': int((similarities > 0).sum()),  # positive = same direction
        'num_negative': int((similarities < 0).sum()),  # negative = opposite direction
        'alignment_rate': float((similarities > 0.5).mean() * 100),  # well-aligned percentage
    }
    
    return similarities, stats


def main():
    parser = argparse.ArgumentParser(description="Compare h and z with detailed analysis")
    parser.add_argument('--model', type=str, required=True, help='Model name (e.g., Qwen/Qwen-7B)')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--layers', type=str, required=True, help='Comma-separated layer numbers (e.g., 4,5,6,7,8)')
    parser.add_argument('--z_method', type=str, default='all', help='Z computation method')
    parser.add_argument('--h_cache_dir', type=str, default='./h_cache', help='H cache directory')
    parser.add_argument('--z_cache_dir', type=str, default='./zs_cache', help='Z cache directory')
    parser.add_argument('--output_dir', type=str, default='./experiments/results', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Parse layers
    layers = [int(x.strip()) for x in args.layers.split(',')]
    layers = sorted(layers)
    last_layer = layers[-1]
    
    print(f"\n{'='*80}")
    print(f"COMPARE H vs Z ANALYSIS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Seed: {args.seed}")
    print(f"Layers: {layers}")
    print(f"Last layer: {last_layer}")
    print(f"Z method: {args.z_method}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Prepare result storage
    results = {
        'metadata': {
            'model': args.model,
            'dataset': args.dataset,
            'seed': args.seed,
            'layers': layers,
            'last_layer': last_layer,
            'z_method': args.z_method,
            'timestamp': timestamp,
        },
        'experiment1_update_ratio': {},  # per layer (incremental effect)
        'experiment1b_update_ratio_cumulative_current': {},  # per layer (cumulative effect on current layer)
        'experiment2_cosine_similarity': {},  # per layer (incremental direction)
        'experiment3_cumulative_effect': {},  # per layer (cumulative effect on last layer)
    }
    
    # Load z for last layer (target)
    print(f"\nLoading z for last layer {last_layer}...")
    z_last = load_z_cache(args.z_cache_dir, args.dataset, args.model, args.z_method, last_layer, args.seed)
    num_samples = z_last.shape[1]
    print(f"Loaded z_last shape: {z_last.shape}, num_samples: {num_samples}")
    
    # ==================================================
    # EXPERIMENT 1: Update Ratio for each layer
    # ==================================================
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1: Update Ratio Analysis")
    print(f"{'='*80}")
    print(f"Metric: ||h_post_i - z_i|| / ||h_pre_i - z_i||")
    print(f"Interpretation: ratio < 1 means getting closer to ideal state")
    
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        
        # Load data
        h_pre = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_pre_current', layer, args.seed)
        h_post = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_post_current', layer, args.seed)
        z_i = load_z_cache(args.z_cache_dir, args.dataset, args.model, args.z_method, layer, args.seed)
        
        print(f"  h_pre shape: {h_pre.shape}")
        print(f"  h_post shape: {h_post.shape}")
        print(f"  z_i shape: {z_i.shape}")
        
        # Compute update ratio
        ratios, stats = compute_update_ratio(h_pre, h_post, z_i)
        
        # Print statistics
        print(f"\n  Update Ratio Statistics:")
        print(f"    Mean:   {stats['mean']:.4f}")
        print(f"    Std:    {stats['std']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Min:    {stats['min']:.4f}")
        print(f"    Max:    {stats['max']:.4f}")
        print(f"    Q25:    {stats['q25']:.4f}")
        print(f"    Q75:    {stats['q75']:.4f}")
        print(f"    Improved samples (ratio < 1): {stats['num_improved']}/{num_samples} ({stats['improvement_rate']:.2f}%)")
        print(f"    Degraded samples (ratio > 1): {stats['num_degraded']}/{num_samples}")
        
        # Store results
        results['experiment1_update_ratio'][f'layer_{layer}'] = {
            'statistics': stats,
            'per_sample_ratios': ratios.tolist(),  # for visualization
        }
    
    # ==================================================
    # EXPERIMENT 1b: Update Ratio (Cumulative - Current Layer)
    # ==================================================
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 1b: Update Ratio - Cumulative Effect on Current Layer")
    print(f"{'='*80}")
    print(f"Metric: ||h_post_i - z_i|| / ||h_pre_original_i - z_i||")
    print(f"Interpretation: ratio < 1 means cumulative editing makes layer i closer to z_i")
    
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        
        # Load data
        h_pre_original = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_pre_original', layer, args.seed)
        h_post = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_post_current', layer, args.seed)
        z_i = load_z_cache(args.z_cache_dir, args.dataset, args.model, args.z_method, layer, args.seed)
        
        print(f"  h_pre_original shape: {h_pre_original.shape}")
        print(f"  h_post shape: {h_post.shape}")
        print(f"  z_i shape: {z_i.shape}")
        
        # Compute update ratio (cumulative)
        ratios, stats = compute_update_ratio(h_pre_original, h_post, z_i)
        
        # Print statistics
        print(f"\n  Cumulative Update Ratio Statistics:")
        print(f"    Mean:   {stats['mean']:.4f}")
        print(f"    Std:    {stats['std']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Min:    {stats['min']:.4f}")
        print(f"    Max:    {stats['max']:.4f}")
        print(f"    Q25:    {stats['q25']:.4f}")
        print(f"    Q75:    {stats['q75']:.4f}")
        print(f"    Improved samples (ratio < 1): {stats['num_improved']}/{num_samples} ({stats['improvement_rate']:.2f}%)")
        print(f"    Degraded samples (ratio > 1): {stats['num_degraded']}/{num_samples}")
        
        # Store results
        results['experiment1b_update_ratio_cumulative_current'][f'layer_{layer}'] = {
            'statistics': stats,
            'per_sample_ratios': ratios.tolist(),
        }
    
    # ==================================================
    # EXPERIMENT 2: Cosine Similarity for each layer
    # ==================================================
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 2: Cosine Similarity Analysis")
    print(f"{'='*80}")
    print(f"Metric: cos(z_last - h_pre_i_last, h_post_i_last - h_pre_i_last)")
    print(f"Interpretation: similarity > 0.5 means well-aligned with ideal direction")
    
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- Layer {layer} ---")
        
        # Load data
        h_pre_last_at_i = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_pre_last_at_layer', layer, args.seed)
        h_post_last_at_i = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_post_last_at_layer', layer, args.seed)
        
        print(f"  h_pre_last_at_{layer} shape: {h_pre_last_at_i.shape}")
        print(f"  h_post_last_at_{layer} shape: {h_post_last_at_i.shape}")
        
        # Compute directions
        ideal_direction = z_last - h_pre_last_at_i  # where we want to go
        edit_direction = h_post_last_at_i - h_pre_last_at_i  # where we actually went
        
        # Compute cosine similarity
        similarities, stats = compute_cosine_similarity(ideal_direction, edit_direction)
        
        # Print statistics
        print(f"\n  Cosine Similarity Statistics:")
        print(f"    Mean:   {stats['mean']:.4f}")
        print(f"    Std:    {stats['std']:.4f}")
        print(f"    Median: {stats['median']:.4f}")
        print(f"    Min:    {stats['min']:.4f}")
        print(f"    Max:    {stats['max']:.4f}")
        print(f"    Q25:    {stats['q25']:.4f}")
        print(f"    Q75:    {stats['q75']:.4f}")
        print(f"    Positive alignment (>0): {stats['num_positive']}/{num_samples}")
        print(f"    Well-aligned (>0.5): {int((similarities > 0.5).sum())}/{num_samples} ({stats['alignment_rate']:.2f}%)")
        print(f"    Negative alignment (<0): {stats['num_negative']}/{num_samples}")
        
        # Store results
        results['experiment2_cosine_similarity'][f'layer_{layer}'] = {
            'statistics': stats,
            'per_sample_similarities': similarities.tolist(),  # for visualization
        }
    
    # ==================================================
    # EXPERIMENT 3: Cumulative Effect on Last Layer
    # ==================================================
    print(f"\n{'='*80}")
    print(f"EXPERIMENT 3: Cumulative Effect on Last Layer")
    print(f"{'='*80}")
    print(f"Evaluates whether cumulative editing progressively moves last layer closer to z_last")
    
    # Load h_pre_original for last layer (use layer[0]'s h_pre_original since it's the original model)
    # Actually, we need h_pre_original[last_layer]
    print(f"\nLoading h_pre_original for last layer {last_layer}...")
    h_pre_original_last = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_pre_original', last_layer, args.seed)
    print(f"h_pre_original_last shape: {h_pre_original_last.shape}")
    
    # Compute ideal direction (fixed for all layers)
    ideal_direction = z_last - h_pre_original_last
    ideal_distance = torch.linalg.norm(ideal_direction, dim=0)  # [num_samples]
    
    print(f"\nIdeal direction: z_last - h_pre_original_last")
    print(f"  Mean norm: {ideal_distance.mean():.4f}")
    
    for layer in tqdm(layers, desc="Processing layers"):
        print(f"\n--- After editing up to Layer {layer} ---")
        
        # Load h_post_last_at_layer (last layer output after editing up to layer i)
        h_post_last_at_i = load_h_cache(args.h_cache_dir, args.dataset, args.model, 'h_post_last_at_layer', layer, args.seed)
        print(f"  h_post_last_at_{layer} shape: {h_post_last_at_i.shape}")
        
        # Metric 1: Distance ratio (how much closer to target?)
        dist_after_edit = torch.linalg.norm(h_post_last_at_i - z_last, dim=0)  # [num_samples]
        dist_ratios = (dist_after_edit / (ideal_distance + 1e-10)).detach().float().numpy()
        
        ratio_stats = {
            'mean': float(dist_ratios.mean()),
            'std': float(dist_ratios.std()),
            'min': float(dist_ratios.min()),
            'max': float(dist_ratios.max()),
            'median': float(np.median(dist_ratios)),
            'q25': float(np.percentile(dist_ratios, 25)),
            'q75': float(np.percentile(dist_ratios, 75)),
            'num_improved': int((dist_ratios < 1.0).sum()),
            'num_degraded': int((dist_ratios > 1.0).sum()),
            'improvement_rate': float((dist_ratios < 1.0).mean() * 100),
        }
        
        # Metric 2: Cosine similarity (is direction aligned?)
        cumulative_direction = h_post_last_at_i - h_pre_original_last
        dot_products = (ideal_direction * cumulative_direction).sum(dim=0)
        norm_ideal = torch.linalg.norm(ideal_direction, dim=0)
        norm_cumulative = torch.linalg.norm(cumulative_direction, dim=0)
        cosine_sims = (dot_products / (norm_ideal * norm_cumulative + 1e-10)).detach().float().numpy()
        
        cosine_stats = {
            'mean': float(cosine_sims.mean()),
            'std': float(cosine_sims.std()),
            'min': float(cosine_sims.min()),
            'max': float(cosine_sims.max()),
            'median': float(np.median(cosine_sims)),
            'q25': float(np.percentile(cosine_sims, 25)),
            'q75': float(np.percentile(cosine_sims, 75)),
            'num_positive': int((cosine_sims > 0).sum()),
            'num_negative': int((cosine_sims < 0).sum()),
            'alignment_rate': float((cosine_sims > 0.5).mean() * 100),
        }
        
        # Print statistics
        print(f"\n  Distance Ratio (closer to z_last?):")
        print(f"    Mean:   {ratio_stats['mean']:.4f}")
        print(f"    Median: {ratio_stats['median']:.4f}")
        print(f"    Improved: {ratio_stats['num_improved']}/{num_samples} ({ratio_stats['improvement_rate']:.2f}%)")
        
        print(f"\n  Direction Alignment (cosine similarity):")
        print(f"    Mean:   {cosine_stats['mean']:.4f}")
        print(f"    Median: {cosine_stats['median']:.4f}")
        print(f"    Well-aligned (>0.5): {int((cosine_sims > 0.5).sum())}/{num_samples} ({cosine_stats['alignment_rate']:.2f}%)")
        
        # Store results
        results['experiment3_cumulative_effect'][f'layer_{layer}'] = {
            'distance_ratio_statistics': ratio_stats,
            'cosine_similarity_statistics': cosine_stats,
            'per_sample_distance_ratios': dist_ratios.tolist(),
            'per_sample_cosine_similarities': cosine_sims.tolist(),
        }
    
    # ==================================================
    # Save results to JSON
    # ==================================================
    output_filename = f"compare_h_z_{args.dataset}_{args.model.replace('/', '-')}_seed{args.seed}_{timestamp}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    print(f"\n{'='*80}")
    print(f"SAVING RESULTS")
    print(f"{'='*80}")
    print(f"Output file: {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved successfully!")
    
    # ==================================================
    # Generate summary report
    # ==================================================
    summary_filename = f"summary_compare_h_z_{args.dataset}_{args.model.replace('/', '-')}_seed{args.seed}_{timestamp}.txt"
    summary_path = os.path.join(args.output_dir, summary_filename)
    
    with open(summary_path, 'w') as f:
        f.write(f"{'='*80}\n")
        f.write(f"COMPARE H vs Z SUMMARY REPORT\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Layers: {layers}\n")
        f.write(f"Num samples: {num_samples}\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"EXPERIMENT 1: Update Ratio (||h_post - z|| / ||h_pre - z||)\n")
        f.write(f"Incremental effect: each layer's individual contribution\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Layer':<10} {'Mean':<10} {'Median':<10} {'Improved%':<12} {'Min':<10} {'Max':<10}\n")
        f.write(f"{'-'*70}\n")
        for layer in layers:
            stats = results['experiment1_update_ratio'][f'layer_{layer}']['statistics']
            f.write(f"{layer:<10} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                   f"{stats['improvement_rate']:<12.2f} {stats['min']:<10.4f} {stats['max']:<10.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"EXPERIMENT 1b: Update Ratio - Cumulative (Current Layer)\n")
        f.write(f"Cumulative effect: compared to original model, is layer i closer to z_i?\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Layer':<10} {'Mean':<10} {'Median':<10} {'Improved%':<12} {'Min':<10} {'Max':<10}\n")
        f.write(f"{'-'*70}\n")
        for layer in layers:
            stats = results['experiment1b_update_ratio_cumulative_current'][f'layer_{layer}']['statistics']
            f.write(f"{layer:<10} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                   f"{stats['improvement_rate']:<12.2f} {stats['min']:<10.4f} {stats['max']:<10.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"EXPERIMENT 2: Cosine Similarity (alignment with ideal direction)\n")
        f.write(f"Incremental direction: each layer's direction contribution\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Layer':<10} {'Mean':<10} {'Median':<10} {'Aligned%':<12} {'Min':<10} {'Max':<10}\n")
        f.write(f"{'-'*70}\n")
        for layer in layers:
            stats = results['experiment2_cosine_similarity'][f'layer_{layer}']['statistics']
            f.write(f"{layer:<10} {stats['mean']:<10.4f} {stats['median']:<10.4f} "
                   f"{stats['alignment_rate']:<12.2f} {stats['min']:<10.4f} {stats['max']:<10.4f}\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"EXPERIMENT 3: Cumulative Effect on Last Layer\n")
        f.write(f"Progressive improvement: is last layer getting closer to z_last?\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"{'Layer':<10} {'DistRatio':<12} {'Improved%':<12} {'Cosine':<12} {'Aligned%':<12}\n")
        f.write(f"{'-'*70}\n")
        for layer in layers:
            dist_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['distance_ratio_statistics']
            cos_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['cosine_similarity_statistics']
            f.write(f"{layer:<10} {dist_stats['mean']:<12.4f} {dist_stats['improvement_rate']:<12.2f} "
                   f"{cos_stats['mean']:<12.4f} {cos_stats['alignment_rate']:<12.2f}\n")
    
    print(f"✓ Summary report saved to: {summary_path}")
    
    # Print final summary to console
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"\nEXPERIMENT 1: Update Ratio (Incremental) - Average across layers")
    avg_improvement_rate = np.mean([
        results['experiment1_update_ratio'][f'layer_{layer}']['statistics']['improvement_rate']
        for layer in layers
    ])
    print(f"  Average improvement rate: {avg_improvement_rate:.2f}%")
    
    print(f"\nEXPERIMENT 1b: Update Ratio (Cumulative - Current Layer) - Average across layers")
    avg_improvement_rate_cumulative = np.mean([
        results['experiment1b_update_ratio_cumulative_current'][f'layer_{layer}']['statistics']['improvement_rate']
        for layer in layers
    ])
    print(f"  Average improvement rate: {avg_improvement_rate_cumulative:.2f}%")
    
    print(f"\nEXPERIMENT 2: Cosine Similarity (Incremental) - Average across layers")
    avg_alignment_rate = np.mean([
        results['experiment2_cosine_similarity'][f'layer_{layer}']['statistics']['alignment_rate']
        for layer in layers
    ])
    print(f"  Average alignment rate: {avg_alignment_rate:.2f}%")
    
    print(f"\nEXPERIMENT 3: Cumulative Effect on Last Layer")
    print(f"  Distance Ratio by Layer (should decrease):")
    for layer in layers:
        dist_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['distance_ratio_statistics']
        print(f"    Layer {layer}: {dist_stats['mean']:.4f} (improved: {dist_stats['improvement_rate']:.1f}%)")
    
    print(f"\n  Direction Alignment by Layer (should stay high):")
    for layer in layers:
        cos_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['cosine_similarity_statistics']
        print(f"    Layer {layer}: {cos_stats['mean']:.4f} (aligned: {cos_stats['alignment_rate']:.1f}%)")
    
    print(f"\n{'='*80}")
    print(f"DONE!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
