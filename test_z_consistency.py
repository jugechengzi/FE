"""
Test if precomputed z and real-time computed z are identical.
"""
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from algs.memit.compute_z import compute_z
from algs.memit.memit_main import get_context_templates
from load import load_data
import numpy as np
import random
import os


def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def test_z_consistency(cfg: DictConfig):
    """
    Test if precomputed z matches real-time computed z.
    Usage: python test_z_consistency.py llms=llama3-8b data=multi_counterfact_20877 seed=0
    """
    print("="*80)
    print("TESTING Z CONSISTENCY")
    print("="*80)
    
    set_random_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"\nLoading model {cfg.llms.name}...")
    if cfg.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.model_dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.model_dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg.llms.name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    tok = AutoTokenizer.from_pretrained(cfg.llms.name, trust_remote_code=True)
    
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id
    
    # Load data
    print(f"\nLoading data {cfg.data}...")
    data = load_data(cfg)
    
    # Get context templates
    print("\nGenerating context templates...")
    context_templates = get_context_templates(model, tok)
    print(f"Context templates: {context_templates}")
    
    # Compute z for first 5 samples
    z_layer = cfg.llms.layers[-1]
    num_test_samples = 5
    
    print(f"\n{'='*80}")
    print(f"Computing z for first {num_test_samples} samples at layer {z_layer}")
    print(f"{'='*80}")
    
    z_realtime_list = []
    for i, request in enumerate(data[:num_test_samples]):
        request_copy = dict(request)
        request_copy["target_new"] = " " + request_copy["target_new"]
        
        cur_z = compute_z(
            model,
            tok,
            request_copy,
            cfg,
            z_layer,
            context_templates,
        )
        z_realtime_list.append(cur_z)
        print(f"Sample {i}: z shape {cur_z.shape}, norm {cur_z.norm():.4f}")
    
    z_realtime = torch.stack(z_realtime_list, dim=1)  # [hidden_dim, num_samples]
    
    # Load precomputed z
    print(f"\n{'='*80}")
    print("Loading precomputed z cache")
    print(f"{'='*80}")
    
    model_cache_name = cfg.llms.name.replace("/", "-")
    dataset_name = cfg.data
    seed_value = cfg.seed
    z_method = "all"
    
    cache_zs_file = f"{cfg.zs_cache_dir}/{dataset_name}-{model_cache_name}-{z_method}-seed{seed_value}-layer{z_layer}.pt"
    
    if not os.path.exists(cache_zs_file):
        print(f"ERROR: Cache file not found: {cache_zs_file}")
        print("Please run precompute_z.py first!")
        return
    
    z_precomputed_full = torch.load(cache_zs_file, map_location='cpu')
    z_precomputed = z_precomputed_full[:, :num_test_samples]
    print(f"Loaded precomputed z: shape {z_precomputed.shape}")
    
    # Compare
    print(f"\n{'='*80}")
    print("COMPARISON RESULTS")
    print(f"{'='*80}")
    
    # Move to same device for comparison
    z_realtime_cpu = z_realtime.cpu()
    z_precomputed_cpu = z_precomputed.cpu()
    
    print(f"\nRealtime z shape: {z_realtime_cpu.shape}")
    print(f"Precomputed z shape: {z_precomputed_cpu.shape}")
    
    # Compute differences
    abs_diff = torch.abs(z_realtime_cpu - z_precomputed_cpu)
    rel_diff = abs_diff / (torch.abs(z_realtime_cpu) + 1e-10)
    
    print(f"\nAbsolute difference:")
    print(f"  Mean: {abs_diff.mean():.6e}")
    print(f"  Max: {abs_diff.max():.6e}")
    print(f"  Min: {abs_diff.min():.6e}")
    
    print(f"\nRelative difference (%):")
    print(f"  Mean: {rel_diff.mean() * 100:.4f}%")
    print(f"  Max: {rel_diff.max() * 100:.4f}%")
    
    # Check if they're close
    is_close = torch.allclose(z_realtime_cpu, z_precomputed_cpu, rtol=1e-5, atol=1e-7)
    print(f"\ntorch.allclose (rtol=1e-5, atol=1e-7): {is_close}")
    
    # Per-sample comparison
    print(f"\n{'='*80}")
    print("PER-SAMPLE COMPARISON")
    print(f"{'='*80}")
    
    for i in range(num_test_samples):
        z_rt = z_realtime_cpu[:, i]
        z_pc = z_precomputed_cpu[:, i]
        
        sample_abs_diff = torch.abs(z_rt - z_pc)
        sample_rel_diff = sample_abs_diff / (torch.abs(z_rt) + 1e-10)
        
        cosine_sim = torch.nn.functional.cosine_similarity(z_rt.unsqueeze(0), z_pc.unsqueeze(0))
        
        print(f"\nSample {i}:")
        print(f"  Realtime norm: {z_rt.norm():.6f}")
        print(f"  Precomputed norm: {z_pc.norm():.6f}")
        print(f"  Abs diff mean: {sample_abs_diff.mean():.6e}")
        print(f"  Abs diff max: {sample_abs_diff.max():.6e}")
        print(f"  Rel diff mean: {sample_rel_diff.mean() * 100:.4f}%")
        print(f"  Cosine similarity: {cosine_sim.item():.8f}")
    
    print(f"\n{'='*80}")
    if is_close:
        print("✓ SUCCESS: Precomputed z and realtime z are IDENTICAL!")
    else:
        print("✗ FAILURE: Precomputed z and realtime z are DIFFERENT!")
        print("This explains the 83% vs 96% EFFICACY difference!")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_z_consistency()
