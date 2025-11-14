import torch
from torch.nn import functional as F

def cross_entropy_between_logits(logits1, logits2):
    if type(logits1) is not torch.Tensor:
        logits1 = torch.tensor(logits1).to('cuda')
    if type(logits2) is not torch.Tensor:
        logits2 = torch.tensor(logits2).to('cuda')
    # 将logits2转换为概率分布
    probs2 = F.softmax(logits2, dim=-1)
    # 计算交叉熵
    return -torch.sum(probs2 * F.log_softmax(logits1, dim=-1))

if __name__ == "__main__":
    result_path = "/home/weiliu/student/xhm/edit-result-final"
    ds_name = ["zsre_mend_eval_19086", "multi_counterfact_20877"]
    # ds_name = "multi_counterfact_20877"
    llm_name = "meta-llama-Meta-Llama-3-8B-Instruct"
    alg_name = ["alphaedit","emmet"]
    for ds in ds_name:
        target_logits_dict = torch.load(f'{result_path}/{ds}/{llm_name}/bs2000-local_cov_neighborhood_target_logits.pt')
        for alg in alg_name:
            edited_logits_dict = torch.load(f'{result_path}/{ds}/{llm_name}/{alg}-bs2000-local_cov_neighborhood_target_logits.pt')
            target_true_logits = target_logits_dict['target_true_logits']
            edited_true_logits = edited_logits_dict['target_true_logits']
            ce_loss = 0
            num_samples = len(target_true_logits)
            for i in range(len(target_true_logits)):
                true_logits = target_true_logits[i]
                edited_logits = edited_true_logits[i]
                cur_ans_len = len(true_logits)
                for j in range(len(true_logits)):
                    ce_loss+=cross_entropy_between_logits(edited_logits[j], true_logits[j]).cpu().item()/cur_ans_len
                    j+=1
                i+=1
            print("Dataset:", ds, "LLM:", llm_name, "Algorithm:", alg)
            print(f'Cross Entropy between edited logits and target logits: {ce_loss/num_samples}')