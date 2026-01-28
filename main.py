import os
import torch
import random
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

from algs.wise import apply_wise_to_model
from evals.evaluation import eval_one_edit
from algs.alphaedit import apply_alphaedit_to_model
from algs.memit import apply_memit_to_model
from algs.rome import apply_rome_to_model
from algs.emmet import apply_emmet_to_model
from algs.rect import apply_rect_to_model
from algs.namet import apply_namet_to_model
from algs.prune import apply_prune_to_model
from algs.pmet import apply_pmet_to_model
from algs.adaedit import apply_adaedit_to_model
from algs.wise import apply_wise_to_model
from algs.ft import apply_ft_to_model
from algs.rledit import apply_rledit_to_model
from datetime import datetime

from load import load_model,load_data,save_model
from util.utility import ensure_file_directory
import numpy as np
import time

ALG_DICT = {
    "alphaedit":  apply_alphaedit_to_model,
    "memit": apply_memit_to_model,
    "memit_nok0": apply_memit_to_model,
    "rome": apply_rome_to_model,
    "emmet": apply_emmet_to_model,
    "namet": apply_namet_to_model,
    "namet_nok0": apply_namet_to_model,
    "rect": apply_rect_to_model,
    "rect_nok0": apply_rect_to_model,
    "prune": apply_prune_to_model,
    "prune_nok0": apply_prune_to_model,
    "pmet": apply_pmet_to_model,
    "pmet_nok0": apply_pmet_to_model,
    "adaedit": apply_adaedit_to_model,
    "adaedit_nok0": apply_adaedit_to_model,
    "wise": apply_wise_to_model,
    "ft": apply_ft_to_model,
    "rledit": apply_rledit_to_model,
}

MODELSCOPE_PATH = "/home/liubingqing/.cache/modelscope/LLM-Research/"
MODEL_PATH = {"meta-llama/Llama-3.1-8B-Instruct": MODELSCOPE_PATH + "Meta-Llama-3___1-8B-Instruct",
              "meta-llama/Llama-3-8B-Instruct": MODELSCOPE_PATH + "Meta-Llama-3-8B-Instruct",
              "meta-llama/Llama-3.2-3B-Instruct": MODELSCOPE_PATH + "Llama-3___2-3B-Instruct", }


def set_random_seed(seed=42):
    torch.manual_seed(seed)  # torch的cpu随机性
    torch.cuda.manual_seed_all(seed)  # torch的gpu随机性
    torch.backends.cudnn.benchmark = False  # 保证gpu每次都选择相同的算法，但是不保证该算法是deterministic的。
    torch.backends.cudnn.deterministic = True  # 紧接着上面，保证算法是deterministic的。
    np.random.seed(seed)  # np的随机性。
    random.seed(seed)  # python的随机性。
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置python哈希种子

import hydra
from omegaconf import DictConfig, OmegaConf


def print_dict(dict):
    for key, value in dict.items():
        print(key, value)

def eval_algo(cfg,model,tok,data):
    all_metrics = {}
    # filtered_data = []
    #一个一个评估。
    for edit in data:#不支持不同的样本有不同的key评估。
        metrics=eval_one_edit(cfg,model,tok,edit)
        if metrics is None:
            continue
        # if metrics.get('rewrite_prompts_correct', 0) == 1:
        #     filtered_data.append(edit)
        if len(all_metrics)==0:
            for key,value in metrics.items():
                all_metrics[key]=[value]
        else:
            for key,value in metrics.items():
                all_metrics[key].append(value)
    # #保存过滤后的数据集。
    # file=cfg.data_dir+"/"+f"{cfg.data}_{cfg.llms.name.replace('/','-')}"+"_filtered.json"
    # ensure_file_directory(file)
    # with open(file, "w", encoding="utf-8") as f:
    #     json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    #对所有指标进行总结。
    avg_metrics={}
    for key,value in all_metrics.items():
        avg_metrics[key]=np.round(np.mean(value),3).item()
    return avg_metrics

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_random_seed(cfg.seed)
    device = torch.device("cuda:{}".format(cfg.gpu) if torch.cuda.is_available() else "cpu")
    print("Start Loading model")
    model_name=cfg.llms.name
    model_name_or_path=MODEL_PATH.get(model_name,model_name)

    if cfg.model_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif cfg.model_dtype == "float16":
        torch_dtype = torch.float16
    elif cfg.model_dtype == "float32":
        torch_dtype = torch.float32
    else:
        torch_dtype = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,torch_dtype=torch_dtype,trust_remote_code=True).to(device)
    tok = AutoTokenizer.from_pretrained(model_name_or_path,trust_remote_code=True)
    print("Loading model successfully")

    # tok.pad_token = tok.eos_token

    if "QWenTokenizer" in str(type(tok)):
        # Qwen 官方定义：<|endoftext|> 是 eos, pad, bos
        # 针对Qwen第一代
        target_token = "<|endoftext|>"
        tok.eos_token = target_token
        tok.pad_token = target_token
        tok.bos_token = target_token

    if tok.pad_token is None:
        try:
            tok.pad_token = tok.eos_token
            tok.pad_token_id = tok.eos_token_id
        except AttributeError:
            # 如果无法直接赋值，尝试直接修改内部变量或属性，针对chatglm等特殊tokenizer
            tok.pad_token_id = tok.eos_token_id


    # 打印确认，方便在日志中排查
    print(f"Tokenizer Setup: Pad Token = {tok.pad_token}, Pad ID = {tok.pad_token_id}, Side = {tok.padding_side}")

    # print(model.transformer.h[8])
    # return

    apply_algo = ALG_DICT[cfg.algs.name]
    data=load_data(cfg)
    if cfg.unlearning_ab:
        if "llama" not in cfg.llms.name.lower():
            raise ValueError("目前只有llama3能进行unleaning_ab_test!")
        from evals.lweval import unleaning_ab_predictions
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/unleaning_ab_predictions_all.txt"

        if not os.path.exists(file):
            print("Start Evaluating the Original Model")
            pre_metrics = eval_algo(cfg, model, tok, data)
            ensure_file_directory(file)
            with open(file, "w", encoding="utf-8") as f:
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
                for line in unleaning_ab_predictions:
                    f.write(line + "\n")
            unleaning_ab_predictions.clear()
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)

        edited_model=load_model(model,cfg)
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/unlearning_ab_predictions_all.txt"
        ensure_file_directory(file)
        with open(file, "w", encoding="utf-8") as f:
            f.write("The Evaluation Results before Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
            for line in unleaning_ab_predictions:
                f.write(line + "\n")
        print("\n\n")
        print("The Evaluation Results after Editing:")
        print_dict(post_metrics)

    if cfg.tf_props:
        from evals.lweval import tf_props
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/tf_props.npy"
        if not os.path.exists(file):
            eval_algo(cfg, model, tok, data)
            ensure_file_directory(file)
            np.save(file,np.array(tf_props))
            tf_props.clear()

        eval_algo(cfg, edited_model, tok, data)
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/tf_props.npy"
        ensure_file_directory(file)
        np.save(file,np.array(tf_props))

    if cfg.inversion_tf:
        if "llama" not in cfg.llms.name.lower() or cfg.data !="multi_counterfact_20877":
            raise ValueError("目前只有llama3在multi_counterfact_20877上才能进行inversion_tf_test!")
        from evals.lweval import inversion_tf_predicts
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/inversion_tf.txt"
        
        with open(cfg.data_dir+"/cf_after_inversion.json", "r") as f:
            data = json.load(f)
        print("data len=" + str(len(data)))
        if not os.path.exists(file):
            print("Start Evaluating the Original Model")
            pre_metrics = eval_algo(cfg, model, tok, data)
            ensure_file_directory(file)
            with open(file, "w", encoding="utf-8") as f:
                for line in inversion_tf_predicts:
                    f.write(line + "\n")
                f.write("\n\n")
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            inversion_tf_predicts.clear()
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)

        edited_model=load_model(model,cfg)
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/inversion_tf.txt"
        ensure_file_directory(file)
        with open(file, "w", encoding="utf-8") as f:
            for line in inversion_tf_predicts:
                f.write(line + "\n")
            f.write("\n\n")
            f.write("The Evaluation Results before Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
        print("\n\n")
        print("The Evaluation Results after Editing:")
        print_dict(post_metrics)

    if cfg.test_only:
        from evals.lweval import predicts
        from evals.lweval import abcd_orders
        pre_results_file = cfg.results_dir + "/{}/{}/{}".format(cfg.data, cfg.llms.name.replace("/", "-"), cfg.save_name)
        ensure_file_directory(pre_results_file)
        if not os.path.exists(pre_results_file):
            start_time = time.time()
            print("Start Evaluating the Original Model")
            pre_metrics=eval_algo(cfg, model, tok, data)#不是每一次都有必要进行这个。
            end_time = time.time()
            hours = np.round((end_time - start_time) / 3600, 3)
            with open(pre_results_file, "w", encoding="utf-8") as f:
                f.write("\n\nEvaluation Took {} Hours".format(hours))
                f.write("\n\n")
                f.write("The Evaluation Results before Editing:")
                f.write("\n\n")
                json.dump(pre_metrics, f, ensure_ascii=False, indent=2)
                f.write("\n\n")
            if cfg.neighborhood_logits:
                if cfg.data == "zsre_mend_eval_19086":
                    from evals.zsre import target_true_logits,target_new_logits
                else:
                    from evals.counterfact import target_true_logits,target_new_logits
                logits_dict = {
                    "target_true_logits": target_true_logits,
                    "target_new_logits": target_new_logits
                }
                torch.save(logits_dict, pre_results_file + "_neighborhood_target_logits.pt")
                target_true_logits.clear()
                target_new_logits.clear()
            print("End Evaluating the Original Model")
            if cfg.lw_eval:
                file=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/pred_lw_eval.npy"
                file_orders=cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-")+"/"+cfg.data+"/abcd_orders.npy"
                ensure_file_directory(file)
                ensure_file_directory(file_orders)
                np.save(file,np.array(predicts))
                np.save(file_orders,np.array(abcd_orders))
                abcd_orders.clear()
                predicts.clear()
        edited_model=load_model(model,cfg)
        print("Start Evaluating the Edited Model")
        post_metrics = eval_algo(cfg, edited_model, tok, data)
        # formatted_time = datetime.now().strftime("%d_%H_%M_%S")
        # post_results_file = cfg.results_dir + "/{}/{}/{}-{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.num_edits,formatted_time)
        post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(cfg.data, cfg.llms.name.replace("/","-"),cfg.algs.name, cfg.save_name)

        ensure_file_directory(post_results_file)
        with open(post_results_file, "w", encoding="utf-8") as f:
            f.write(OmegaConf.to_yaml(cfg) + "\n\n")  # 写入字符串，加空行分隔
            f.write("The Evaluation Results after Editing:")
            f.write("\n\n")
            json.dump(post_metrics, f, ensure_ascii=False, indent=2)
            f.write("\n\n")
        print("End Evaluating the Edited Model")

        # write to local csv file
        csv_file = cfg.results_dir + "/summary_results.csv"
        file_exists = os.path.isfile(csv_file)
        with open(csv_file, "a", encoding="utf-8") as f:
            if not file_exists:
                # write header
                headers = ['model', 'algo', 'data', 'v_lr', 'v_num_grad_steps', 'num_edits'] + list(post_metrics.keys())
                f.write(",".join(headers) + "\n")
            
            # only write metrics with v_lr, v_num_grad_steps, num_edits, model_name, alg_name
            values = [cfg.llms.name, cfg.algs.name, cfg.data, str(cfg.llms.get('v_lr', 'N/A')), str(cfg.llms.get('v_num_grad_steps', 'N/A')), str(cfg.get('num_edits', 'N/A'))]
            # No add timestamp to csv
            values += [str(post_metrics[key]) for key in post_metrics.keys()]
            f.write(",".join(values) + "\n")

        if cfg.neighborhood_logits:
            if cfg.data == "zsre_mend_eval_19086":
                from evals.zsre import target_true_logits,target_new_logits
            else:
                from evals.counterfact import target_true_logits,target_new_logits
            logits_dict = {
                "target_true_logits": target_true_logits,
                "target_new_logits": target_new_logits
            }
            torch.save(logits_dict, post_results_file + "_neighborhood_target_logits.pt")
        if cfg.lw_eval:
            file = cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/pred_lw_eval.npy"
            file_orders = cfg.results_dir+"/lw_eval/"+cfg.llms.name.replace("/", "-") + "/" + cfg.data+"/"+cfg.algs.name + "/abcd_orders.npy"
            ensure_file_directory(file)
            ensure_file_directory(file_orders)
            np.save(file, np.array(predicts))
            np.save(file_orders,np.array(abcd_orders))
    elif not cfg.tf_props and not cfg.inversion_tf and not cfg.unlearning_ab:
        if cfg.debug_mode:
            pre_metrics = eval_algo(cfg, model, tok, data)  # 不是每一次都有必要进行这个。

        edited_model = apply_algo(model,tok,data,cfg)

        if cfg.debug_mode:
            post_metrics = eval_algo(cfg, edited_model, tok, data)
            print("The Evaluation Results before Editing:")
            print_dict(pre_metrics)
            print("\n\n")
            print("The Evaluation Results after Editing:")
            print_dict(post_metrics)
        else:
            save_model(edited_model,cfg)
    
    # write evaluation output to txt under the same folder as model saving


if __name__ == "__main__":
    main()
