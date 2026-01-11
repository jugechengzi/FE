"""
Analyze distribution of experimental results from compare_h_z.py output.

Provides detailed statistics including:
- Mean and standard deviation for each layer
- Distribution across bins (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0, >1.0)
"""

import json
import argparse
import numpy as np
from pathlib import Path


def analyze_distribution(values, bins):
    """
    Analyze distribution of values across bins.
    
    Args:
        values: list or array of values
        bins: list of bin edges [0, 0.2, 0.4, 0.6, 0.8, 1.0, inf]
    
    Returns:
        dict with bin counts and percentages
    """
    values = np.array(values)
    counts = {}
    
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        
        if upper == float('inf'):
            mask = values >= lower
            label = f"≥{lower}"
        else:
            mask = (values >= lower) & (values < upper)
            label = f"[{lower}, {upper})"
        
        count = int(mask.sum())
        percentage = (count / len(values)) * 100 if len(values) > 0 else 0
        counts[label] = {'count': count, 'percentage': percentage}
    
    return counts


def print_experiment_stats(experiment_name, results_dict, layers, bins, value_key='per_sample_ratios'):
    """Print statistics for a single experiment."""
    print(f"\n{'='*100}")
    print(f"{experiment_name}")
    print(f"{'='*100}")
    
    for layer in layers:
        layer_key = f'layer_{layer}'
        if layer_key not in results_dict:
            print(f"\n⚠ Layer {layer} not found in results")
            continue
        
        stats = results_dict[layer_key]['statistics']
        values = results_dict[layer_key][value_key]
        
        print(f"\n{'─'*100}")
        print(f"LAYER {layer}")
        print(f"{'─'*100}")
        
        # Basic statistics
        print(f"\n📊 Basic Statistics:")
        print(f"  Mean:     {stats['mean']:8.4f}")
        print(f"  Std:      {stats['std']:8.4f}")
        print(f"  Median:   {stats['median']:8.4f}")
        print(f"  Min:      {stats['min']:8.4f}")
        print(f"  Max:      {stats['max']:8.4f}")
        print(f"  Q25:      {stats['q25']:8.4f}")
        print(f"  Q75:      {stats['q75']:8.4f}")
        
        # Distribution across bins
        dist = analyze_distribution(values, bins)
        print(f"\n📈 Distribution:")
        print(f"  {'Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
        print(f"  {'-'*70}")
        
        for bin_label, bin_data in dist.items():
            count = bin_data['count']
            percentage = bin_data['percentage']
            bar_length = int(percentage / 2)  # Scale to fit terminal
            bar = '█' * bar_length
            print(f"  {bin_label:<15} {count:<10} {percentage:>6.2f}%       {bar}")


def print_cumulative_experiment_stats(experiment_name, results_dict, layers, 
                                      ratio_bins, cosine_bins):
    """Print statistics for experiment 3 which has both ratio and cosine."""
    print(f"\n{'='*100}")
    print(f"{experiment_name}")
    print(f"{'='*100}")
    
    for layer in layers:
        layer_key = f'layer_{layer}'
        if layer_key not in results_dict:
            print(f"\n⚠ Layer {layer} not found in results")
            continue
        
        ratio_stats = results_dict[layer_key]['distance_ratio_statistics']
        ratio_values = results_dict[layer_key]['per_sample_distance_ratios']
        
        cosine_stats = results_dict[layer_key]['cosine_similarity_statistics']
        cosine_values = results_dict[layer_key]['per_sample_cosine_similarities']
        
        print(f"\n{'─'*100}")
        print(f"LAYER {layer}")
        print(f"{'─'*100}")
        
        # Distance Ratio Statistics
        print(f"\n📊 Distance Ratio Statistics:")
        print(f"  Mean:     {ratio_stats['mean']:8.4f}")
        print(f"  Std:      {ratio_stats['std']:8.4f}")
        print(f"  Median:   {ratio_stats['median']:8.4f}")
        print(f"  Min:      {ratio_stats['min']:8.4f}")
        print(f"  Max:      {ratio_stats['max']:8.4f}")
        print(f"  Improved: {ratio_stats['num_improved']} ({ratio_stats['improvement_rate']:.2f}%)")
        
        dist = analyze_distribution(ratio_values, ratio_bins)
        print(f"\n📈 Distance Ratio Distribution:")
        print(f"  {'Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
        print(f"  {'-'*70}")
        
        for bin_label, bin_data in dist.items():
            count = bin_data['count']
            percentage = bin_data['percentage']
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"  {bin_label:<15} {count:<10} {percentage:>6.2f}%       {bar}")
        
        # Cosine Similarity Statistics
        print(f"\n📊 Cosine Similarity Statistics:")
        print(f"  Mean:     {cosine_stats['mean']:8.4f}")
        print(f"  Std:      {cosine_stats['std']:8.4f}")
        print(f"  Median:   {cosine_stats['median']:8.4f}")
        print(f"  Min:      {cosine_stats['min']:8.4f}")
        print(f"  Max:      {cosine_stats['max']:8.4f}")
        print(f"  Aligned:  {int((np.array(cosine_values) > 0.5).sum())} ({cosine_stats['alignment_rate']:.2f}%)")
        
        dist = analyze_distribution(cosine_values, cosine_bins)
        print(f"\n📈 Cosine Similarity Distribution:")
        print(f"  {'Range':<15} {'Count':<10} {'Percentage':<15} {'Bar'}")
        print(f"  {'-'*70}")
        
        for bin_label, bin_data in dist.items():
            count = bin_data['count']
            percentage = bin_data['percentage']
            bar_length = int(percentage / 2)
            bar = '█' * bar_length
            print(f"  {bin_label:<15} {count:<10} {percentage:>6.2f}%       {bar}")


def main():
    parser = argparse.ArgumentParser(description="Analyze distribution of compare_h_z results")
    parser.add_argument('--result_file', type=str, required=True, 
                       help='Path to compare_h_z JSON result file')
    
    args = parser.parse_args()
    
    # Load results
    print(f"\n{'='*100}")
    print(f"LOADING RESULTS")
    print(f"{'='*100}")
    print(f"File: {args.result_file}")
    
    with open(args.result_file, 'r') as f:
        results = json.load(f)
    
    # Extract metadata
    metadata = results['metadata']
    layers = metadata['layers']
    
    print(f"\nMetadata:")
    print(f"  Model:    {metadata['model']}")
    print(f"  Dataset:  {metadata['dataset']}")
    print(f"  Seed:     {metadata['seed']}")
    print(f"  Layers:   {layers}")
    print(f"  Z method: {metadata['z_method']}")
    
    # Define bins
    ratio_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
    cosine_bins = [-1.0, -0.5, 0, 0.2, 0.4, 0.6, 0.8, 1.0, float('inf')]
    
    # Analyze Experiment 1: Update Ratio (Incremental)
    print_experiment_stats(
        "EXPERIMENT 1: Update Ratio - Incremental Effect",
        results['experiment1_update_ratio'],
        layers,
        ratio_bins,
        value_key='per_sample_ratios'
    )
    
    # # Analyze Experiment 1b: Update Ratio (Cumulative - Current Layer)
    # print_experiment_stats(
    #     "EXPERIMENT 1b: Update Ratio - Cumulative Effect (Current Layer)",
    #     results['experiment1b_update_ratio_cumulative_current'],
    #     layers,
    #     ratio_bins,
    #     value_key='per_sample_ratios'
    # )
    
    # Analyze Experiment 2: Cosine Similarity (Incremental)
    print_experiment_stats(
        "EXPERIMENT 2: Cosine Similarity - Incremental Direction",
        results['experiment2_cosine_similarity'],
        layers,
        cosine_bins,
        value_key='per_sample_similarities'
    )
    
    # # Analyze Experiment 3: Cumulative Effect on Last Layer
    # print_cumulative_experiment_stats(
    #     "EXPERIMENT 3: Cumulative Effect on Last Layer",
    #     results['experiment3_cumulative_effect'],
    #     layers,
    #     ratio_bins,
    #     cosine_bins
    # )
    
    # Summary across all layers
    print(f"\n{'='*100}")
    print(f"CROSS-LAYER SUMMARY")
    print(f"{'='*100}")
    
    print(f"\n📊 Experiment 1 (Incremental) - Mean Ratio by Layer:")
    for layer in layers:
        stats = results['experiment1_update_ratio'][f'layer_{layer}']['statistics']
        print(f"  Layer {layer}: {stats['mean']:.4f}")
    
    # print(f"\n📊 Experiment 1b (Cumulative) - Average Improvement Rate by Layer:")
    # for layer in layers:
    #     stats = results['experiment1b_update_ratio_cumulative_current'][f'layer_{layer}']['statistics']
    #     print(f"  Layer {layer}: {stats['improvement_rate']:>6.2f}% (mean ratio: {stats['mean']:.4f})")
    
    print(f"\n📊 Experiment 2 (Incremental) - Mean Cosine Similarity by Layer:")
    for layer in layers:
        stats = results['experiment2_cosine_similarity'][f'layer_{layer}']['statistics']
        print(f"  Layer {layer}: {stats['mean']:.4f}")
    
    # print(f"\n📊 Experiment 3 (Cumulative) - Distance Ratio Trend:")
    # for layer in layers:
    #     dist_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['distance_ratio_statistics']
    #     print(f"  After Layer {layer}: {dist_stats['mean']:.4f} ({dist_stats['improvement_rate']:.2f}% improved)")
    # 
    # print(f"\n📊 Experiment 3 (Cumulative) - Direction Alignment Trend:")
    # for layer in layers:
    #     cos_stats = results['experiment3_cumulative_effect'][f'layer_{layer}']['cosine_similarity_statistics']
    #     print(f"  After Layer {layer}: {cos_stats['mean']:.4f} ({cos_stats['alignment_rate']:.2f}% aligned)")
    
    print(f"\n{'='*100}")
    print(f"ANALYSIS COMPLETE")
    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
