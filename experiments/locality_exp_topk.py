import hydra
import torch
import torch.nn.functional as F

def get_topk_overlap_count(logits1, logits2, k):
    if not torch.is_tensor(logits1):
        logits1 = torch.tensor(logits1)
        logits2 = torch.tensor(logits2)
    logits1 = logits1.to("cuda")
    logits2 = logits2.to("cuda")
    _, topk1 = torch.topk(logits1, k, dim=-1)
    _, topk2 = torch.topk(logits2, k, dim=-1)
    topk1_extended = topk1.unsqueeze(-1)
    topk2_extended = topk2.unsqueeze(-2)
    equality_matrix = (topk1_extended == topk2_extended)
    overlap_matrix = equality_matrix.any(dim=-1)
    count = overlap_matrix.sum(dim=-1)
    return count


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    data = "multi_counterfact_20877"
    data = "zsre_mend_eval_19086"

    # model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    print("Evaluating locality for model {}".format(cfg.llms.name))

    # algs = ["adaedit","alphaedit","emmet","namet","pmet","prune","rect"]
    algs = ["wise"]
    # algs = ['namet']
    # algs = ["alphaedit"]
    # algs = [ "prune", "rect" ]

    print("Evaluating method {}".format(algs))

    # save_names = [f"{x}cov-bs2000-neighborhood-logits" for x in [ "2500","5000", "7500", "10000", "12500", "17500", "20000", "22500", "25000", "27500", "30000"]]
    # save_names = [f"{x}cov-bs2000-neighborhood-logits" for x in ["1.5e2","1.5e3", "1.5e4","1.5e5", "1.5e6"]]
    save_names = [f"{y}-{x}cov-bs2000-neighborhood-logits" for y in ["t"] for x in [ "1.5e4"]]
    # save_names = [f"{x}-{_lambda}-bs2000-neighborhood-logits" for x in ["z","t","r","blue","a"] for _lambda in ["1.5e4"]]
    # save_names = [f"a-{x}e4-bs2000-vlr0.1-steps80-wd0.5-kl0.0625-clamp0.75-neighborhood-logits" for x in ["0.5","1","1.5"]]


    # save_names = [f"a-1e4-batch2000-vlr0.1-steps80-neighborhood-logits",f"a-1e4-batch2000-vlr0.05-steps100-neighborhood-logits"]
    # save_names = [f"a-1e4-batch2000-vlr0.05-steps100-neighborhood-logits",f"a-1e4-batch2000-vlr0.03-steps120-neighborhood-logits"]


    print("Evaluating dataset: {}".format(data))

    for alg in algs:
        print("\n")
        print("Evaluating locality for method {}".format(alg))
        for save_name in save_names:
            pre_results_file = cfg.results_dir + "/{}/{}/{}".format(data, cfg.llms.name.replace("/", "-"), save_name)
            ori_logits_dict_filename = pre_results_file + "_neighborhood_target_logits.pt"
            ori_logits = torch.load(ori_logits_dict_filename)["target_true_logits"]
            
            post_results_file = cfg.results_dir + "/{}/{}/{}-{}".format(data, cfg.llms.name.replace("/","-"),alg, save_name)
            post_logits_dict_filename = post_results_file + "_neighborhood_target_logits.pt"
            post_logits = torch.load(post_logits_dict_filename)["target_true_logits"]
            print("Post result file path: "+post_logits_dict_filename)

            total_num = len(ori_logits)
            count_1 = 0
            count_5 = 0
            count_10 = 0
            for i in range(total_num):
                ori_y = ori_logits[i][0]
                post_y = post_logits[i][0]
                count_1 += get_topk_overlap_count(ori_y, post_y, 1)
                count_5 += get_topk_overlap_count(ori_y, post_y, 5)
                count_10 += get_topk_overlap_count(ori_y, post_y, 10)
            print(save_name+" top-1 rate: "+ str(count_1/total_num))
            print(save_name+" top-5 rate: "+ str(count_5/(total_num*5)))
            print(save_name+" top-10 rate: "+ str(count_10/(total_num*10)))
        print("\n")


if __name__ == "__main__":
    main()