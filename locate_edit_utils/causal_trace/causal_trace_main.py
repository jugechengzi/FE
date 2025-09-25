import argparse
import json
import os
import re
from collections import defaultdict

import numpy
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from locate_edit_utils.causal_trace.knowns import KnownsDataset
from locate_edit_utils.tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)
from util import nethook
from util.runningstats import Covariance, tally

DATA_DIR="/home/liubingqing/project/MI/data"
def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match("^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_name",
        default="gpt2-xl",
        # choices=[
        #     "gpt2-xl",
        #     "EleutherAI/gpt-j-6B",
        #     "EleutherAI/gpt-neox-20b",
        #     "gpt2-large",
        #     "gpt2-medium",
        #     "gpt2",
        # ],
    )
    aa("--fact_file", default=None)
    aa("--output_dir", default="results/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--gpu", default=0, type=int)
    aa("--layers_select", default=0, type=int)

    args = parser.parse_args()

    modeldir = f'r{args.replace}_{args.model_name.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir
    output_dir = args.output_dir.format(model_name=modeldir)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Half precision to let the 20b model fit.
    torch_dtype = torch.float16 if "20b" in args.model_name else None

    mt = ModelAndTokenizer(args.model_name, torch_dtype=torch_dtype,gpu=args.gpu)

    if args.fact_file is None:
        knowns = KnownsDataset(DATA_DIR)#这个就是知识元组，一个列表。
    else:
        with open(args.fact_file) as f:
            knowns = json.load(f)

    knowns.data=knowns.data[:2]#只要10个知识，不要太多。原本是有1209个，运行要特别久。
    noise_level = args.noise_level#s3，暂时不知道什么意思。
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt, [k["subject"] for k in knowns]
            )#mt是模型，第二个参数是三元组的头实体。
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level == "m":
            # Automatic multivariate gaussian
            noise_level = collect_embedding_gaussian(mt)
            print(f"Using multivariate gaussian to match model noise")
        elif noise_level.startswith("t"):
            # Automatic d-distribution with d degrees of freedom
            degrees = float(noise_level[1:])
            noise_level = collect_embedding_tdist(mt, degrees)
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    for knowledge in tqdm(knowns):#取出一个三元组。
        known_id = knowledge["known_id"]
        if args.layers_select ==0:
            kind_ttype=[None, "mlp", "attn"]
        else:
            kind_ttype=[None]
        for kind in kind_ttype:
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            if not os.path.isfile(filename):#上面这个文件就是用来保存，对于某一个知识，各个状态被恢复后模型输出的概率的。保存下来，以后可以不需要再计算。
                result = calculate_hidden_flow(
                    mt,#模型和分词器。
                    knowledge["prompt"],#模型输入。
                    knowledge["subject"],#头实体。
                    expect=knowledge["attribute"],#尾实体，或者说叫做希望的输出。
                    kind=kind,
                    noise=noise_level,#这个是头实体嵌入的方差。
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                )
                numpy_result = {
                    k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                # numpy.savez(filename, **numpy_result)#为每一条知识都搞一个文件?
            else:
                numpy_result = numpy.load(filename, allow_pickle=True)
            if not numpy_result["correct_prediction"]:#给定的数据集好像不毁坏的情况下都是预测正确尾实体的。所以这个if语句不成立。
                tqdm.write(f"Skipping {knowledge['prompt']}")
                continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{str(numpy_result["answer"]).strip()}_{known_id}{kind_suffix}.pdf'
            if known_id > 200:
                continue
            # plot_trace_heatmap(plot_result, savepdf=pdfname)
            plot_trace_heatmap(plot_result, savepdf=False)#不希望保存，只是想看一眼即可。
            if kind==None:
                scores = plot_result["scores"]
                last_id = plot_result["subject_range"][-1] - 1
                print("The recommended layer to be edited (in descending order):")
                print(scores[last_id, :].argsort()[::-1])



def trace_with_patch(
    mt,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    model=mt.model
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)#这是要使用标准正太随机采样，

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:#好像明白了这个states_to_patch是什么意思了，好像就是l层的第t个位置的那个神经元保持正确的意思。
        patch_spec[l].append(t)#但是对于mlp和attn的激活值，第t个位置可能会有多个l，也就是window，10个，不知道是不是求平均作为中心激活值的effect。

    embed_layername = layername(mt.config, 0, "embed")#得到模型第0层的名字。其实就是嵌入曾。transformer.wte,这个是token embedding，不包括position embedding。

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:#x[11,10,1600即[bs,seqlen,hdim]
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:#这个就是头实体的开头和结尾id。
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))#[10,4,1600]，这是因为11个里面有一个不需要加噪音，另外非头实体也不需要加噪音。
                ).to(x.device)#0,1标准正态分布的随机采样。
                if replace:#这个noise_data改成了以0为均值，头实体token嵌入标准差为标注差的正太分布。
                    x[1:, b:e] = noise_data#直接变成噪音。因为标准差差不多所以想要以假乱真？不太可能吧。
                else:
                    x[1:, b:e] += noise_data#加上噪音。第0个输入不要加噪音，作为对照组。
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)#修改h应该就是改变x，对的，的确会。
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]#这个看不懂呀，为什么要改t这个位置的向量。
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,#这个参数说明一定会有嵌入层，但是后面的层不一定。
        edit_output=patch_rep,#这里开始要编辑输出了。
    ) as td:#这个tracedict就是大号的trace，每一个想要处理的模块对应了一个trace，即hook。
        outputs_exp = model(**inp)#这个是加了噪音的之后的输出。不一定，也有可能没有加噪音，有参数选项。

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]
    #这个是得到破坏了头实体token嵌入之后，前向传播输出，然后得到最后一个位置的概率，但是会对所有毁坏样本进行概率平均，也就是说破坏情况下，平均意义下预测没破坏时预测的那个token的概率。一般来说肯定减小了嘛，显著减少。
    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def trace_with_repatch(
    mt,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    model=mt.model
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(mt.config, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def calculate_hidden_flow(
    mt,
    prompt,
    subject,
    samples=10,
    noise=0.1,
    token_range=None,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
    expect=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1),device=mt.model.device)#又是制作输入，就是得到输入token的id。对于一个输入，要复制11次，好像是因为要破坏输入，做对比。
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])#上面是得到预测的下一个token id以及对应的归一化概率。
    if expect is not None and answer.strip() != expect:#根据token id得到对应的token。
        return dict(correct_prediction=False)
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)#分词器，输入token id，头实体文本。
    if token_range == "subject_last":
        token_range = [e_range[1] - 1]
    elif token_range is not None:
        raise ValueError(f"Unknown token_range: {token_range}")
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise, uniform_noise=uniform_noise
    ).item()#破坏输入之后对于原本预测单词的概率的影响。显然概率会减小，原本是9%，现在是1%。
    if not kind:
        differences = trace_important_states(#这个函数根据名字来说，应该是最重要的了。
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            token_range=token_range,
        )
    else:#当kind为mlp或者attn的时候会进入到这个里面。
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            token_range=token_range,
        )
    differences = differences.detach().cpu()#这个就是各个位置各个层的状态进行恢复之后得到的概率结果。[10,48]
    return dict(
        scores=differences,#这个是带恢复的毁坏。
        low_score=low_score,#这个是完全毁坏。
        high_score=base_score,#这个是不毁坏的结果。
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):
    model=mt.model
    ntoks = inp["input_ids"].shape[1]#输入token序列，1当然就是输入长度。
    table = []

    if token_range is None:
        token_range = range(ntoks)
    for tnum in token_range:
        row = []
        for layer in range(num_layers):
            r = trace_with_patch(#又是这个函数，
                mt,
                inp,
                [(tnum, layername(mt.config, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    mt,
    num_layers,
    inp,
    e_range,
    answer_t,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    token_range=None,
):#这个函数还是得好好研究一下，一直没有看明白为什么研究mlp和attn的激活的时候，要进入这个函数。
    model=mt.model
    config=mt.config
    ntoks = inp["input_ids"].shape[1]#输入序列的长度。
    table = []

    if token_range is None:
        token_range = range(ntoks)#这个暂时不知道有什么用。
    for tnum in token_range:
        row = []
        for layer in range(num_layers):#锁定了行，这里又锁定了列。
            layerlist = [
                (tnum, layername(config, L, kind))#这里L会有多个，所以是锁定行，有多个列。
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))#这个max/min都很好理解，毕竟L的含义是层
                )#然后这个range是在干嘛呢？其实就是以某行某列为中心，按照行的维度，向前找5个列，向后找5个列，为什么现在还不知道，或许这就是函数名字window的含义吧。
            ]
            r = trace_with_patch(
                mt,
                inp,
                layerlist,
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )#这个r其实就是对layerlist里面的layer恢复了隐状态之后的输出结果。
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
            gpu=0,
    ):
        from main import MODEL_PATH
        import yaml

        # 打开并加载 YAML 文件
        with open('../../configs/llms/{}.yaml'.format(model_name), 'r', encoding='utf-8') as file:
            self.config = yaml.safe_load(file)  # 推荐使用 safe_load，更安全
        model_name=self.config["name"]
        model_name_or_path = MODEL_PATH.get(model_name, model_name)
        self.device=torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if model is None:
            assert model_name is not None
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
            ).to(self.device)
            nethook.set_requires_grad(False, model)
            model.eval()#好像据说需要30多个GB的GPU，根本搞不定。
            # model.eval().to(torch.device("cpu"))
        self.tokenizer = tokenizer
        self.model = model
        # self.layer_names = [
        #     n
        #     for n, m in model.named_modules()
        #     if (re.match(r"^(transformer|gpt_neox)\.(h|layers)\.\d+$", n))
        # ]#|表示的是或者的意思。^表示的是匹配开头。\d是数字。上面这个其实就是想要得到block，所以开头结尾的嵌入层和
        # self.num_layers = len(self.layer_names)
        self.num_layers =self.model.config.num_hidden_layers
    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(config, num, kind=None):
    if kind=="embed":
        return config["emb_module"]
    elif kind=="mlp":
        return config["mlp_module_tmp"].format(num)
    elif kind=="attn":
        return config["attn_module_tmp"].format(num)
    elif kind==None:
        return config["layer_module_tmp"].format(num)

def guess_subject(prompt):
    return re.search(r"(?!Wh(o|at|ere|en|ich|y) )([A-Z]\S*)(\s[A-Z][a-z']*)*", prompt)[
        0
    ].strip()


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    uniform_noise=False,
    window=10,
    kind=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt,
        prompt,
        subject,
        samples=samples,
        noise=noise,
        uniform_noise=uniform_noise,
        window=window,
        kind=kind,
    )
    plot_trace_heatmap(result, savepdf)


def plot_trace_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    answer = result["answer"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)#result是一个字典。get是取默认值。
    labels = list(result["input_tokens"])
    for i in range(*result["subject_range"]):#这个默认了subject的token必须是相邻的。
        labels[i] = labels[i] + "*"#这个加上*号是为了画图，即标记这些是头实体。

    # with plt.rc_context(rc={"font.family": "Times New Roman"}):#创建一种即时的绘画环境。
    with plt.rc_context(rc={"font.family": "DejaVu Serif"}):#创建一种即时的绘画环境。
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "attn": "Reds"}[
                kind
            ],#不同的隐状态类型使用不同的颜色映射，注意这里使用的是连续映射。还有，这里None重复了吧。
            vmin=low_score,#这个可以保证colorbar从vmin开始映射。
        )
        ax.invert_yaxis()#反转y轴，一般的话，y轴下面是负数，上面是正数，现在就是反过来了。
        ax.set_yticks([0.5 + i for i in range(len(differences))])#y轴刻度所需要的坐标。
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])#
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)#将刻度换成标签。
        if not modelname:
            modelname = "GPT"#按照道理，应该是有标签才对啊。
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")#一个是title和label，title只有一个，而xy轴都有label。
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)#返回colorbar
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:#这个就是正确答案，也就是下一个单词。
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            cb.ax.set_title(f"p({str(answer).strip()})", y=-0.16, fontsize=10)#在colorbar下面写上正确答案。
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_all_flow(mt, prompt, subject=None):
    for kind in ["mlp", "attn", None]:
        plot_hidden_flow(mt, prompt, subject, kind=kind)


# Utilities for dealing with tokens
def make_inputs(tokenizer, prompts, device="cpu"):#把这个就改了，原本的设备是cpu。
    token_lists = [tokenizer.encode(p) for p in prompts]#给定一个头实体，对其进行分词，分词器不一定就是一个单词一个token，有可能一个单词对应几个。
    maxlen = max(len(t) for t in token_lists)#这是计算batch内头实体token的最大长度。
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]#显然，这个是填充。而且将填充的token放到序列的前面。
    # position_ids = [[0] * (maxlen - len(t)) + list(range(len(t))) for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]#0表示填充，1表示真实。
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        #    position_ids=torch.tensor(position_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_token_range(tokenizer, token_array, substring):
    toks = decode_tokens(tokenizer, token_array)#又是解码，应该就是将token id返回到文本吧。不过文本经过分词器再返回到文本，好像不一样了，有些单词会拆分为了多个token。
    whole_string = "".join(toks)#上面说了分词前后的文本可能不一样，但是现在会一样。
    char_loc = whole_string.index(substring)#定位一下头实体在哪。
    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > char_loc:
            tok_start = i
        if tok_end is None and loc >= char_loc + len(substring):
            tok_end = i + 1
            break
    return (tok_start, tok_end)#这里搞了一大堆，无非就是得到头实体在输入token id中的开始位置和结束位置，这其实默认了，构造提示的时候，头实体不可以拆分，应该作为一个专有名词，整体。


def predict_token(mt, prompts, return_p=False):
    inp = make_inputs(mt.tokenizer, prompts,mt.model.device)
    preds, p = predict_from_input(mt.model, inp)
    result = [mt.tokenizer.decode(c) for c in preds]
    if return_p:
        result = (result, p)
    return result


def predict_from_input(model, inp):
    out = model(**inp)["logits"]#inp就是输入，由token id和attention mask构成，形状都是[bs,seqlen],输出是一个字典，他这个model是causallm，输出不一样，logits才是我们要的输出[bs,seqlen,vocab_size]
    probs = torch.softmax(out[:, -1], dim=1)#显然，这个是要最后一个token的输出，然后归一化得到概率。
    p, preds = torch.max(probs, dim=1)#这个是进行预测，得到预测概率和预测id，但是只预测一个token吗？
    return preds, p#形状都是[bs]


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:#一个头实体。
        inp = make_inputs(mt.tokenizer, [s],device=mt.model.device)#这个是把头实体分词为token,得到的id序列。
        with nethook.Trace(mt.model, layername(mt.config, 0, "embed")) as t:
            mt.model(**inp)#这个是开始前向传播了。输入就是token id和填充的mask情况。我这个在cpu上传播，没想到时间还挺久的。
            alldata.append(t.output[0])#这个就是存储指定层的输出。
    alldata = torch.cat(alldata)#上面的速度有点太慢了，以后不搞那么多了，相当于有1209个前向传播。
    noise_level = alldata.std().item()
    return noise_level#这个alldata似乎不用管，因为并不保存，只是为了得到std，看一下这一层输出的方差？


def get_embedding_cov(mt):
    model = mt.model
    tokenizer = mt.tokenizer

    def get_ds():
        ds_name = "wikitext"
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name],
        )
        try:
            maxlen = model.config.n_positions
        except:
            maxlen = 100  # Hack due to missing setting in GPT2-NeoX.
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    ds = get_ds()
    sample_size = 1000
    batch_size = 5
    filename = None
    batch_tokens = 100

    progress = lambda x, **k: x

    stat = Covariance()
    loader = tally(
        stat,
        ds,
        cache=filename,
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=0,
    )
    with torch.no_grad():
        for batch_group in loader:
            for batch in batch_group:
                batch = dict_to_(batch, torch.device("cpu"))
                del batch["position_ids"]
                with nethook.Trace(model, layername(mt.config, 0, "embed")) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                stat.add(feats.cpu().double())
    return stat.mean(), stat.covariance()


def make_generator_transform(mean=None, cov=None):
    d = len(mean) if mean is not None else len(cov)
    device = mean.device if mean is not None else cov.device
    layer = torch.nn.Linear(d, d, dtype=torch.double)
    nethook.set_requires_grad(False, layer)
    layer.to(device)
    layer.bias[...] = 0 if mean is None else mean
    if cov is None:
        layer.weight[...] = torch.eye(d).to(device)
    else:
        _, s, v = cov.svd()
        w = s.sqrt()[None, :] * v
        layer.weight[...] = w
    return layer


def collect_embedding_gaussian(mt):
    m, c = get_embedding_cov(mt)
    return make_generator_transform(m, c)


def collect_embedding_tdist(mt, degree=3):
    # We will sample sqrt(degree / u) * sample, where u is from the chi2[degree] dist.
    # And this will give us variance is (degree / degree - 2) * cov.
    # Therefore if we want to match the sample variance, we should
    # reduce cov by a factor of (degree - 2) / degree.
    # In other words we should be sampling sqrt(degree - 2 / u) * sample.
    u_sample = torch.from_numpy(
        numpy.random.RandomState(2).chisquare(df=degree, size=1000)
    )
    fixed_sample = ((degree - 2) / u_sample).sqrt()
    mvg = collect_embedding_gaussian(mt)

    def normal_to_student(x):
        gauss = mvg(x)
        size = gauss.shape[:-1].numel()
        factor = fixed_sample[:size].reshape(gauss.shape[:-1] + (1,))
        student = factor * gauss
        return student

    return normal_to_student


if __name__ == "__main__":
    main()
