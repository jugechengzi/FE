from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import hydra

import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))

from util import nethook
from locate_edit_utils.layer_stats import get_cov

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype=torch.float32

def get_upd_matrix_by_key(model, cache_dir, alg_name, load_name, llm_name, data):
    weights_dir = cache_dir + "/saved_weights"
    weights_file = weights_dir + "/{}/{}-{}-{}.pt".format(alg_name, data, load_name,
                                                        llm_name.replace("/", "-"))
    weights = torch.load(weights_file, map_location="cpu")
    upd_by_key = {}
    with torch.no_grad():
        for key, value in weights.items():
            weight = nethook.get_parameter(model, key)
            upd_by_key[key] = value.to(dtype).to(device) - weight
    return upd_by_key


def _trace_delta_cov_delta_t(delta: torch.Tensor, cov: torch.Tensor, *, name: str) -> torch.Tensor:
    """Returns tr(Δ Σ Δᵀ), handling possible Δ transpose."""
    if delta.ndim != 2 or cov.ndim != 2:
        raise ValueError(f"Expected 2D tensors for delta/cov, got {delta.shape=} {cov.shape=} ({name})")
    if cov.shape[0] != cov.shape[1]:
        raise ValueError(f"Expected square covariance, got {cov.shape=} ({name})")

    # For c_proj: weight/delta is typically [d_model, 4*d_model], cov is [4*d_model, 4*d_model]
    if delta.shape[1] == cov.shape[0]:
        d = delta
    elif delta.shape[0] == cov.shape[0]:
        d = delta.T
    else:
        raise ValueError(
            f"Delta/cov shapes incompatible for tr(ΔΣΔᵀ): {delta.shape=} {cov.shape=} ({name})"
        )
    return torch.trace(d @ cov @ d.T)

covs = []
def load_cov(cfg,model,tok,layers):
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
            cache_filename_suffix="",
            random_sample=1
        )
        covs.append(cov)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):

    model_name = cfg.llms.name
    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "Qwen/Qwen-7B"

    # load_path = "/mnt/hdd/weiliu/student/xhm/edit-cache-final"
    load_path = "/scratch/hkliu/cache/edit-cache-final"
    print("Load model from {}".format(model_name))

    # multi_counterfact_20877,zsre_mend_eval_19086,"wiki_cf_2266","mquake_cf_9218",
    datasets = ["zsre_mend_eval_19086"]
    # datasets = ["multi_counterfact_20877"]
    algs = ["memit"]

    # _lambda = "1.5e4"
    _lambda = "1.5e2"
    # load_name = f"{_lambda}cov-bs2000"

    # a:传统memit t:新方法 z: baseline只更新第八层
    load_name = f"t-{_lambda}-bs2000"
    
    print("Load edited weights from {}".format(load_name))
    print("Using datasets: {}".format(datasets))
    print("Using algorithms: {}".format(algs))

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    tok = AutoTokenizer.from_pretrained(model_name)
    load_cov(cfg, model, tok, cfg.llms.layers)
    for data in datasets:
        print("================Processing dataset: {}================".format(data))
        for alg in algs:
            print("**********Processing algorithm: {}**********".format(alg))
            # load edited weights with k0k0t
            ori_upd_by_key = get_upd_matrix_by_key(model, load_path, alg, load_name, model_name, data)
            nok0_upd_matrixs = None
            # if alg in ["memit", "adaedit", "namet", "prune", "rect", "pmet"]:
            #     # load edited weights without k0k0t
            #     nok0_upd_matrixs = get_upd_matrix(model, load_path, alg+"_nok0", "bs2000-local_cov", model_name, data)
            for i, layer in enumerate(cfg.llms.layers):
                cov = covs[i].to(dtype).to(device)
                key = f"{cfg.llms.rewrite_module_tmp.format(layer)}.weight"
                if key not in ori_upd_by_key:
                    available = [k for k in ori_upd_by_key.keys() if cfg.llms.rewrite_module_tmp.format(layer) in k]
                    raise KeyError(
                        f"Missing update for {key}. Available keys for this layer: {available[:10]}"
                    )
                ori_delta = ori_upd_by_key[key]
                ori_fnorm = _trace_delta_cov_delta_t(ori_delta, cov, name=key)
                if nok0_upd_matrixs is not None:
                    nok0_delta_mm_cov = nok0_upd_matrixs[i] @ cov @ nok0_upd_matrixs[i].T
                    nok0_fnorm = torch.trace(nok0_delta_mm_cov)
                    print(f">>>>>>Layer: {layer}<<<<<<<\n Frobenius Norm of [ori_delta @ k0]: {ori_fnorm}\n Frobenius Norm of [nok0_delta @ k0]: {nok0_fnorm}")
                else:
                    print(f">>>>>>Layer: {layer}<<<<<<<\n Frobenius Norm of [ori_delta @ k0]: {ori_fnorm}\n No nok0 results available.")

if __name__ == "__main__":
    main()