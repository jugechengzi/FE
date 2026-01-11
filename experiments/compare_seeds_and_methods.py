"""
Systematic comparison of z vectors across seeds and methods.
This script compares:
1. Same method, different seeds (measure stability)
2. Different methods, same seed (measure method difference)
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from itertools import combinations


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        v1 = v1.float()
        v2 = v2.float()
        
        dot_product = torch.dot(v1, v2)
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return (dot_product / (norm1 * norm2)).item()
    else:
        raise ValueError("Both vectors must be 1D")


def normalized_diff(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Calculate ||v1 - v2|| / (||v1|| + ||v2||)."""
    if len(v1.shape) == 1 and len(v2.shape) == 1:
        v1 = v1.float()
        v2 = v2.float()
        
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
    """Load z cache from file."""
    if not Path(cache_path).exists():
        raise FileNotFoundError(f"Cache file not found: {cache_path}")
    
    zs = torch.load(cache_path, map_location=device)
    print(f"Loaded: {cache_path}")
    print(f"  Shape: {zs.shape}")
    return zs


def compare_two_caches(
    z1: torch.Tensor,
    z2: torch.Tensor,
    name1: str,
    name2: str
) -> Dict:
    """Compare two z caches and return statistics with per-sample details."""
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")
    
    num_samples = z1.shape[1]
    
    cos_sims = []
    norm_diffs = []
    per_sample_results = {}
    
    for idx in range(num_samples):
        v1 = z1[:, idx]
        v2 = z2[:, idx]
        
        cos_sim = cosine_similarity(v1, v2)
        norm_diff = normalized_diff(v1, v2)
        
        cos_sims.append(cos_sim)
        norm_diffs.append(norm_diff)
        
        # Save per-sample results
        per_sample_results[str(idx)] = {
            "cosine_similarity": float(cos_sim),
            "normalized_difference": float(norm_diff)
        }
    
    return {
        "comparison": f"{name1} vs {name2}",
        "name1": name1,
        "name2": name2,
        "num_samples": num_samples,
        "statistics": {
            "cosine_similarity": {
                "mean": float(np.mean(cos_sims)),
                "std": float(np.std(cos_sims)),
                "min": float(np.min(cos_sims)),
                "max": float(np.max(cos_sims)),
                "median": float(np.median(cos_sims)),
            },
            "normalized_difference": {
                "mean": float(np.mean(norm_diffs)),
                "std": float(np.std(norm_diffs)),
                "min": float(np.min(norm_diffs)),
                "max": float(np.max(norm_diffs)),
                "median": float(np.median(norm_diffs)),
            }
        },
        "per_sample_results": per_sample_results
    }


def compare_seeds_and_methods(
    cache_dir: str,
    dataset: str,
    model_name: str,
    methods: List[str],
    seeds: List[int],
    layer: int,
    device: str = 'cpu'
) -> Dict:
    """
    Systematic comparison of z vectors.
    
    Args:
        cache_dir: Directory containing z caches
        dataset: Dataset name (e.g., 'multi_counterfact_20877')
        model_name: Model name (e.g., 'meta-llama-Meta-Llama-3-8B-Instruct')
        methods: List of methods (e.g., ['all', 'firstforward'])
        seeds: List of seeds (e.g., [0, 42, 123])
        layer: Layer number
        device: Device to load tensors on
    
    Returns:
        Dictionary with all comparison results
    """
    cache_dir = Path(cache_dir)
    
    # Load all z caches
    print(f"\n{'='*80}")
    print(f"Loading z caches...")
    print(f"{'='*80}\n")
    
    z_caches = {}
    for method in methods:
        for seed in seeds:
            # Construct cache path: {dataset}-{model}-{method}-seed{seed}-layer{layer}.pt
            cache_filename = f"{dataset}-{model_name}-{method}-seed{seed}-layer{layer}.pt"
            cache_path = cache_dir / cache_filename
            
            key = f"{method}_seed{seed}"
            try:
                z_caches[key] = load_z_cache(str(cache_path), device=device)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
    
    if not z_caches:
        raise RuntimeError("No z caches loaded successfully!")
    
    results = {
        "config": {
            "cache_dir": str(cache_dir),
            "dataset": dataset,
            "model_name": model_name,
            "methods": methods,
            "seeds": seeds,
            "layer": layer,
        },
        "comparisons": {}
    }
    
    # 1. Compare same method, different seeds (stability check)
    print(f"\n{'='*80}")
    print(f"PART 1: Same Method, Different Seeds (Stability)")
    print(f"{'='*80}\n")
    
    results["comparisons"]["same_method_different_seeds"] = {}
    
    for method in methods:
        method_seeds = [seed for seed in seeds if f"{method}_seed{seed}" in z_caches]
        
        if len(method_seeds) < 2:
            print(f"Skipping {method}: need at least 2 seeds")
            continue
        
        results["comparisons"]["same_method_different_seeds"][method] = []
        
        # Compare all pairs of seeds for this method
        for seed1, seed2 in combinations(method_seeds, 2):
            key1 = f"{method}_seed{seed1}"
            key2 = f"{method}_seed{seed2}"
            
            print(f"Comparing {key1} vs {key2}...")
            comparison = compare_two_caches(
                z_caches[key1],
                z_caches[key2],
                key1,
                key2
            )
            results["comparisons"]["same_method_different_seeds"][method].append(comparison)
            
            print(f"  Cosine Sim: {comparison['statistics']['cosine_similarity']['mean']:.6f} ± {comparison['statistics']['cosine_similarity']['std']:.6f}")
            print(f"  Norm Diff:  {comparison['statistics']['normalized_difference']['mean']:.6f} ± {comparison['statistics']['normalized_difference']['std']:.6f}")
            print()
            print()
    
    # 2. Compare different methods, same seed (method difference check)
    print(f"\n{'='*80}")
    print(f"PART 2: Different Methods, Same Seed (Method Difference)")
    print(f"{'='*80}\n")
    
    results["comparisons"]["different_methods_same_seed"] = {}
    
    for seed in seeds:
        seed_methods = [method for method in methods if f"{method}_seed{seed}" in z_caches]
        
        if len(seed_methods) < 2:
            print(f"Skipping seed {seed}: need at least 2 methods")
            continue
        
        results["comparisons"]["different_methods_same_seed"][f"seed{seed}"] = []
        
        # Compare all pairs of methods for this seed
        for method1, method2 in combinations(seed_methods, 2):
            key1 = f"{method1}_seed{seed}"
            key2 = f"{method2}_seed{seed}"
            
            print(f"Comparing {key1} vs {key2}...")
            comparison = compare_two_caches(
                z_caches[key1],
                z_caches[key2],
                key1,
                key2
            )
            results["comparisons"]["different_methods_same_seed"][f"seed{seed}"].append(comparison)
            
            print(f"  Cosine Sim: {comparison['statistics']['cosine_similarity']['mean']:.6f} ± {comparison['statistics']['cosine_similarity']['std']:.6f}")
            print(f"  Norm Diff:  {comparison['statistics']['normalized_difference']['mean']:.6f} ± {comparison['statistics']['normalized_difference']['std']:.6f}")
            print()
    
    # 3. Summary statistics
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}\n")
    
    summary = {
        "same_method_different_seeds": {},
        "different_methods_same_seed": {}
    }
    
    # Summarize same method comparisons
    for method, comparisons in results["comparisons"]["same_method_different_seeds"].items():
        cos_sims = [c["statistics"]["cosine_similarity"]["mean"] for c in comparisons]
        norm_diffs = [c["statistics"]["normalized_difference"]["mean"] for c in comparisons]
        
        summary["same_method_different_seeds"][method] = {
            "num_comparisons": len(comparisons),
            "cosine_similarity_avg": float(np.mean(cos_sims)),
            "normalized_difference_avg": float(np.mean(norm_diffs)),
        }
        
        print(f"{method} (across seeds):")
        print(f"  Avg Cosine Sim: {summary['same_method_different_seeds'][method]['cosine_similarity_avg']:.6f}")
        print(f"  Avg Norm Diff:  {summary['same_method_different_seeds'][method]['normalized_difference_avg']:.6f}")
    
    print()
    
    # Summarize different method comparisons
    for seed_key, comparisons in results["comparisons"]["different_methods_same_seed"].items():
        cos_sims = [c["statistics"]["cosine_similarity"]["mean"] for c in comparisons]
        norm_diffs = [c["statistics"]["normalized_difference"]["mean"] for c in comparisons]
        
        summary["different_methods_same_seed"][seed_key] = {
            "num_comparisons": len(comparisons),
            "cosine_similarity_avg": float(np.mean(cos_sims)),
            "normalized_difference_avg": float(np.mean(norm_diffs)),
        }
        
        print(f"{seed_key} (across methods):")
        print(f"  Avg Cosine Sim: {summary['different_methods_same_seed'][seed_key]['cosine_similarity_avg']:.6f}")
        print(f"  Avg Norm Diff:  {summary['different_methods_same_seed'][seed_key]['normalized_difference_avg']:.6f}")
    
    results["summary"] = summary
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare z vectors across seeds and methods",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/scratch/hkliu/cache/edit-cache-final/zs_cache",
        help="Directory containing z caches"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name (e.g., multi_counterfact_20877)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., meta-llama-Meta-Llama-3-8B-Instruct)"
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["all", "firstforward"],
        help="Methods to compare"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 42, 123],
        help="Seeds to compare"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer number"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (cpu/cuda)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: auto-generated)"
    )
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_seeds_and_methods(
        cache_dir=args.cache_dir,
        dataset=args.dataset,
        model_name=args.model_name,
        methods=args.methods,
        seeds=args.seeds,
        layer=args.layer,
        device=args.device
    )
    
    # Save results
    if args.output is None:
        output_dir = Path("./experiments/results/zs_compare")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        methods_str = "-".join(sorted(args.methods))
        seeds_str = "-".join(map(str, sorted(args.seeds)))
        output_path = output_dir / f"compare_seeds_methods_{args.dataset}_{args.model_name}_methods-{methods_str}_seeds-{seeds_str}_layer{args.layer}.json"
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
