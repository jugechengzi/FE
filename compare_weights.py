import torch
cfg = {
    "gpu": 0,
    "cache_dir": "/home/weiliu/student/xhm/edit-cache-final",
    "alg_name": "prune",
    "data": "multi_counterfact_20877",
    "positive_load_name": ["seed1","seed2","seed3"],
    "negetive_load_name": ["bf16_seed1_negetive_edit"],
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct"
}

def print_dict(dict):
    for key, value in dict.items():
        print(key, value)

def print_list(o):
    for i in o:
        print(i)
print("=="*20)
print_dict(cfg)
print("=="*20)
weights_dir = cfg["cache_dir"] + "/saved_weights"
weights_files_p = [weights_dir + "/{}/{}-{}-{}.pt".format(cfg["alg_name"], cfg["data"], load_name,
                                                          cfg["model_name"].replace("/", "-")) for load_name in cfg["positive_load_name"]]

weights_files_n = [weights_dir + "/{}/{}-{}-{}.pt".format(cfg["alg_name"], cfg["data"], load_name,
                                                          cfg["model_name"].replace("/", "-")) for load_name in cfg["negetive_load_name"]]

weights_p=[torch.load(w) for w in weights_files_p]
weights_n=[torch.load(w) for w in weights_files_n]
norm_list_p = [[torch.linalg.norm(w.float(), ord=2) for w in m.values()] for m in weights_p]
norm_list_n = [[torch.linalg.norm(w.float(), ord=2) for w in m.values()] for m in weights_n]

diff_norm_list = [[torch.linalg.norm((m1[k]-m2[k]).float(), ord=2) for k in m1.keys()] for m1 in weights_p for m2 in weights_n]
diff_norm_list_2 = [[torch.linalg.norm((weights_p[i][k]-weights_p[(i+1)%3][k]).float(),ord=2) for k in weights_p[i].keys()] for i in range(3)]

print("正编辑后各层的norm")
print_list(norm_list_p)
print("=="*20)
print("反编辑后各层的norm")
print_list(norm_list_n)
print("=="*20)
print("正反编辑各层矩阵的差的norm")
print_list(diff_norm_list)
print("=="*20)
print("正编辑各层矩阵的差的norm")
print_list(diff_norm_list_2)