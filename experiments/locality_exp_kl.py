import hydra
import torch
import torch.nn.functional as F

def kl_divergence_torch(p, q, device="cpu"):
    """
    使用PyTorch计算KL散度
    """
    # 确保是概率分布
    if not torch.is_tensor(p):
        p = torch.tensor(p)
        q = torch.tensor(q)
    p = p.to(device)
    q = q.to(device)
    p = torch.softmax(p, dim=-1) if p.dim() > 1 else F.softmax(p, dim=0)
    q = torch.softmax(q, dim=-1) if q.dim() > 1 else F.softmax(q, dim=0)
    
    # 避免log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon
    
    # 计算KL散度
    kl = torch.sum(p * torch.log(p / q))
    return kl.item()

def js_divergence_torch(p, q, device="cpu"):
    """
    使用 PyTorch 计算 JS 散度
    p, q: logits（tensor）
    """
    # 确保在同一设备
    if not torch.is_tensor(p):
        p = torch.tensor(p)
        q = torch.tensor(q)
    p = p.to(device)
    q = q.to(device)

    # 转为概率分布
    p = torch.softmax(p, dim=-1) if p.dim() > 1 else F.softmax(p, dim=0)
    q = torch.softmax(q, dim=-1) if q.dim() > 1 else F.softmax(q, dim=0)

    # 避免 log(0)
    epsilon = 1e-10
    p = p + epsilon
    q = q + epsilon

    # 混合分布 M
    m = 0.5 * (p + q)

    # JS = 1/2 KL(p||m) + 1/2 KL(q||m)
    kl_pm = torch.sum(p * torch.log(p / m))
    kl_qm = torch.sum(q * torch.log(q / m))

    js = 0.5 * (kl_pm + kl_qm)
    return js.item()

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    data = "multi_counterfact_20877"
    # data = "zsre_mend_eval_19086"

    print("Evaluating locality for model {}".format(cfg.llms.name))
    # save_names = [f"{x}cov-bs2000-neighborhood-logits" for x in ["2500","5000", "7500", "10000", "12500", "17500", "20000", "22500", "25000", "27500", "30000"]]
    # save_names  = [f"{x}cov-bs2000-neighborhood-logits" for x in ["1.5e2","1.5e3","1.5e4","1.5e5", "1.5e6"]]
    # save_names = [f"{x}cov-bs2000-neighborhood-logits" for x in ["1.5e4"]]


    # algs = ["wise", "rledit","memit","adaedit","alphaedit","emmet","namet","pmet","prune","rect"]

    
    algs = ["memit"]
    save_names = [f"{x}-{_lambda}-bs2000-neighborhood-logits" for x in ["z","t","a"] for _lambda in ["1.5e4"]]


    print ("Evaluating method {}".format(algs))
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

            total_num = len(ori_logits)
            kl_divergence_mean = 0
            js_divergence_mean = 0
            for i in range(total_num):
                ori_y = ori_logits[i][0]
                post_y = post_logits[i][0]
                kl_divergence_mean += kl_divergence_torch(ori_y, post_y, "cuda:0") / total_num
                js_divergence_mean += js_divergence_torch(ori_y, post_y, "cuda:0") / total_num
            print(save_name+" kl_divergence_mean: "+str(kl_divergence_mean))
            print(save_name+" js_divergence_mean: "+str(js_divergence_mean))
        print("\n")


if __name__ == "__main__":
    main()