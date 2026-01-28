"""
This is the version of NAMET that applies new methods for computing the target zs.
The precompute zs method cannot be used here because NAMET itself is already a new method to calculate the target zs. It changes the padding side of the tokenizer to add noise and get more robust zs.
To apply new methods for computing the target zs in NAMET, this file tries to directly compute the results without precomputing and saving to cache.
The difference is that in the original NAMET, the zs are computed for the last layer only while in this version, the zs are computed for the first layer with new padding side and then propagated to the last layer.
The only problem is that we do not kwon whether the propagation needs to change the padding side again or not. Here we simply change the padding side during propagation, keeping it consistent before computing the zs.
"""
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        covs.append(cov)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def get_fc_dim(model,cfg):
    W_out = nethook.get_parameter(model, f"{cfg.llms.rewrite_module_tmp.format(1)}.weight")
    fc_dim=W_out.shape[0] if W_out.shape[0]>W_out.shape[1] else W_out.shape[1]
    return fc_dim

def apply_namet_to_model(
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
        requests[i]["target_new"] = " " + request["target_new"]
    layers=cfg.llms.layers
    #查看KKT是否已经计算好。
    for i, layer in enumerate(layers):
        Cpathi = cfg.cache_dir + "/stats/"+ cfg.llms.name.replace("/","-") + "/layer-" + str(layer) +("-" if cfg.cache_filename_suffix !="" else "")+ cfg.cache_filename_suffix + ".npz"
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
    for requests_chunks in chunks(requests, cfg.bs):
        batch_edit(cfg,model,tok,requests_chunks,device,cache_c)
    return model

def batch_edit(cfg, model, tok, requests, device, cache_c):
    # deltas = {}
    # Retrieve weights that user desires to change
    weights = {
        f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in cfg.llms.layers
    }
    # Compute z for final layer
    context_templates = get_context_templates(model, tok)
    layers_sorted = sorted(cfg.llms.layers)
    z_layer = layers_sorted[0]  # First layer
    
    z_list = []

    tok.padding_side = "left"  # Change padding side to add noise
    print(f"Set tokenizer padding side to {tok.padding_side} for more robust z computation.")
    print("Computing zs for the first layer...\n")
    for request in requests:
        #add_noise to obtain more robust value by simply changing the padding_side。
        
        cur_z = compute_z(
            model,
            tok,
            request,
            cfg,
            z_layer,
            context_templates,
        )
        
        z_list.append(cur_z)
    # zs = torch.stack(z_list, dim=1)#[dim,bs]
    z_first = torch.stack(z_list, dim=1)  # [dim, bs]
    zs_dict = {z_layer: z_first}

    # Propagate zs to the last layer
    print("Propagating zs to the last layer...")
    other_layers = layers_sorted[1:]  # Exclude first layer

    for sample_idx, request in enumerate(requests):
        
        # Forward propagate through layers
        prompt = request["prompt"].format(request["subject"])
        input_tok = tok(prompt, return_tensors="pt").to(device)
        
        # Find lookup index
        fact_token = cfg.llms.fact_token
        lookup_idx = find_fact_lookup_idx(request["prompt"], request["subject"], tok, fact_token, verbose=False)
        
        def edit_output_fn(cur_out, cur_layer_name):
            first_layer_module = cfg.llms.layer_module_tmp.format(z_layer)
            # 仅在第一层需要修改的层处注入z
            if cur_layer_name == first_layer_module:
                z_to_inject = zs_dict[z_layer][:, sample_idx]
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
    
    zs = zs_dict

    print("\nFinished computing zs for all layers. Setting tokenizer padding side back to right.")
    tok.padding_side = "right"  # Change back to original padding side


    for i, layer in enumerate(cfg.llms.layers):
        print(f"\n\nLAYER {layer}\n")
        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, cfg, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        if cfg.negetive_prompt_test:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                # z_layer,
                layer, # current layer, not the last one
                context_templates=[request["negetive_prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        else:
            # Compute residual error
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                # z_layer,
                layer, # current layer, not the last one
                context_templates=[request["prompt"] for request in requests],
                words=[request["subject"] for request in requests],
                module_template=cfg.llms.layer_module_tmp,
                fact_token_strategy=cfg.llms.fact_token,
            )[1].T
        targets = zs[layer] - cur_zs #[dim,bs]
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        # resid = targets / (len(cfg.llms.layers) - i)  # Distribute residual across layers
        resid = targets  # Do not distribute residual across layers

        cov = covs[i]
        upd_type = torch.float32
        resid = resid.to(upd_type) # Ensure resid is in the correct dtype, otherwise there may be dtype mismatch and causes errors.

        if cfg.algs.L2 != 0:
            upd_matrix = torch.linalg.solve(
                layer_ks @ layer_ks.T + cache_c[i, :, :].to(device)+
                cfg.algs.L2 * torch.eye(layer_ks.shape[0], dtype=upd_type, device=device),
                layer_ks.to(upd_type) @ resid.T,
            )
        else:
            coef=cfg.llms.mom2_update_weight[i]
            upd_matrix = torch.linalg.solve(
                layer_ks @ layer_ks.T + cache_c[i, :, :].to(device)+coef*cov.to(device)+
                cfg.algs.L2 * torch.eye(layer_ks.shape[0], dtype=upd_type, device=device),
                layer_ks.to(upd_type) @ resid.T,
            )
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
