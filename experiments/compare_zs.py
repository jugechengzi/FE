"""
Compare z vectors from different computation methods.
Computes cosine similarity and normalized difference for each request/idx.
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns value between -1 and 1.
    Automatically converts to float32 to handle dtype mismatches.
    """
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        # Convert to float32 to handle dtype mismatches (float32, float16, bfloat16, etc.)
        v1 = v1.float()
        v2 = v2.float()
        
        # v1, v2 shape: [dim]
        dot_product = torch.dot(v1, v2)
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    else:
        raise ValueError("Both vectors must be 1D")


def normalized_diff(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """
    Calculate ||v1 - v2|| / (||v1|| + ||v2||).
    Normalized difference between two vectors.
    Automatically converts to float32 to handle dtype mismatches.
    """
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        # Convert to float32 to handle dtype mismatches (float32, float16, bfloat16, etc.)
        v1 = v1.float()
        v2 = v2.float()
        
        # v1, v2 shape: [dim]
        diff = v1 - v2
        diff_norm = torch.norm(diff)
        
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        norm_sum = norm1 + norm2
        
        if norm_sum == 0:
            return 0.0
        
        return (diff_norm / norm_sum).item()
    else:
        raise ValueError("Both vectors must be 1D")


def load_z_cache(cache_path: str, device: str = 'cpu') -> torch.Tensor:
    """
    Load z cache from file.
    Expected shape: [dim, num_samples]
    """
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    zs = torch.load(cache_path, map_location=device)
    print(f"Loaded z cache from {cache_path}")
    print(f"  Shape: {zs.shape}")
    return zs


def compare_z_methods(
    cache_path_1: str,
    cache_path_2: str,
    method_name_1: str,
    method_name_2: str,
    device: str = 'cpu'
) -> Dict:
    """
    Compare two z caches.
    
    Args:
        cache_path_1: Path to first z cache file
        cache_path_2: Path to second z cache file
        method_name_1: Name of first method
        method_name_2: Name of second method
        device: Device to load tensors on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with comparison results for each idx
    """
    # Load z caches
    print(f"\nLoading z caches...")
    z1 = load_z_cache(cache_path_1, device=device)
    z2 = load_z_cache(cache_path_2, device=device)
    
    # Validate shapes
    if z1.shape[0] != z2.shape[0]:
        raise ValueError(f"Dimension mismatch: z1 has dim {z1.shape[0]}, z2 has dim {z2.shape[0]}")
    
    if z1.shape[1] != z2.shape[1]:
        raise ValueError(f"Sample count mismatch: z1 has {z1.shape[1]} samples, z2 has {z2.shape[1]} samples")
    
    num_samples = z1.shape[1]
    print(f"Number of samples to compare: {num_samples}")
    
    # Compare each sample
    print(f"\nComputing metrics for each request/idx...")
    results = {
        "config": {
            "method_1": method_name_1,
            "method_2": method_name_2,
            "num_samples": num_samples,
        },
        "results": {}
    }
    
    for idx in range(num_samples):
        v1 = z1[:, idx]
        v2 = z2[:, idx]
        
        cos_sim = cosine_similarity(v1, v2)
        norm_diff = normalized_diff(v1, v2)
        
        results["results"][str(idx)] = {
            "cosine_similarity": cos_sim,
            "normalized_difference": norm_diff,
        }
        
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{num_samples} samples")
    
    # Compute aggregate statistics
    cos_sims = [v["cosine_similarity"] for v in results["results"].values()]
    norm_diffs = [v["normalized_difference"] for v in results["results"].values()]
    
    results["statistics"] = {
        "cosine_similarity": {
            "mean": float(np.mean(cos_sims)),
            "std": float(np.std(cos_sims)),
            "min": float(np.min(cos_sims)),
            "max": float(np.max(cos_sims)),
        },
        "normalized_difference": {
            "mean": float(np.mean(norm_diffs)),
            "std": float(np.std(norm_diffs)),
            "min": float(np.min(norm_diffs)),
            "max": float(np.max(norm_diffs)),
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare z vectors from different methods")
    
    # Cache paths
    parser.add_argument(
        "--cache1",
        type=str,
        required=True,
        help="Path to first z cache file"
    )
    parser.add_argument(
        "--cache2",
        type=str,
        required=True,
        help="Path to second z cache file"
    )
    
    # Method names
    parser.add_argument(
        "--method1",
        type=str,
        default="method_1",
        help="Name of first z method"
    )
    parser.add_argument(
        "--method2",
        type=str,
        default="method_2",
        help="Name of second z method"
    )
    
    # Dataset and other identifiers
    parser.add_argument(
        "--dataset",
        type=str,
        default="multi_counterfact",
        help="Dataset name"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama-Meta-Llama-3-8B-Instruct",
        help="Model name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=4,
        help="Layer number"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path. If not specified, uses default naming."
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help="Device to load tensors on"
    )
    
    args = parser.parse_args()
    
    # Generate output path if not specified
    if args.output is None:
        output_filename = (
            f"compare_zs_{args.dataset}_{args.model.replace('/', '-')}_"
            f"{args.method1}_vs_{args.method2}_"
            f"seed{args.seed}_layer{args.layer}.json"
        )
        output_path = Path(__file__).parent / output_filename
    else:
        output_path = Path(args.output)
    
    print("=" * 80)
    print("Z Vector Comparison")
    print("=" * 80)
    print(f"Cache 1: {args.cache1}")
    print(f"Cache 2: {args.cache2}")
    print(f"Method 1: {args.method1}")
    print(f"Method 2: {args.method2}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print(f"Layer: {args.layer}")
    print(f"Device: {args.device}")
    print(f"Output: {output_path}")
    print("=" * 80)
    
    # Run comparison
    results = compare_z_methods(
        cache_path_1=args.cache1,
        cache_path_2=args.cache2,
        method_name_1=args.method1,
        method_name_2=args.method2,
        device=args.device
    )
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Statistics Summary")
    print("=" * 80)
    stats = results["statistics"]
    print(f"\nCosine Similarity:")
    print(f"  Mean:  {stats['cosine_similarity']['mean']:.6f}")
    print(f"  Std:   {stats['cosine_similarity']['std']:.6f}")
    print(f"  Min:   {stats['cosine_similarity']['min']:.6f}")
    print(f"  Max:   {stats['cosine_similarity']['max']:.6f}")
    
    print(f"\nNormalized Difference:")
    print(f"  Mean:  {stats['normalized_difference']['mean']:.6f}")
    print(f"  Std:   {stats['normalized_difference']['std']:.6f}")
    print(f"  Min:   {stats['normalized_difference']['min']:.6f}")
    print(f"  Max:   {stats['normalized_difference']['max']:.6f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
