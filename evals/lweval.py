import typing
import numpy as np
import scipy
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from util.generate import generate_fast
import random
predicts=[]
abcd_orders=[]
tf_props=[]
inversion_tf_predicts=[]
unleaning_ab_predictions=[]
def check_next_word_binary(texts, target_words,class1="true",class2="false", class3=None, class4=None):
    predictions=[]
    for i in range(len(texts)):
        text=texts[i]
        target_word=target_words[i]
        words = text.lower().split()  # 转小写便于匹配
        # target_word = target_word.lower()
        # if target_word == len(words):#说明续写的词和answer黏在一起了。只有这种情况了，不可能target word大于后者，是出错的情况。
        #     target_word=target_word-1这个其实就是下面这种情况。
        target_word=min(len(words)-1,target_word)#永远不会越界。
        if class1 == words[target_word]:
            predictions.append(0)
        elif class2 == words[target_word]:
            predictions.append(1)
        elif class3 != None and class3 == words[target_word]:
            predictions.append(2)
        elif class4 != None and class4 == words[target_word]:
            predictions.append(3)
        else:
            predictions.append(-1)
    return predictions


# def build_prompt(cfg,prompt):
#     # new_prompt=prompt
#     default_tf_prompt_template=("According to your knowledge, decide whether the "
#                              "following hypothesis is true or false. "
#                              "Only answer with “True” or “False”. "
#                              "Hypothesis: {} Answer:")
#     if cfg.llms.name=="meta-llama/Llama-3-8B-Instruct":
#         new_prompt=build_llama38b_tf_prompt(prompt)
#     else:
#         new_prompt=default_tf_prompt_template.format(prompt)
#     return new_prompt
def build_qwen_tf_prompt(prompt):
    qwen_prompt=(f"<|im_start|>system\nYou are a strict fact-checker. Judge whether the following statement is factually correct based only on your knowledge.\n"
                 f"Respond with exactly one word: either 'True' or 'False'. Do not provide explanations.\n<|im_end|>"
    f"<|im_start|>user\n{prompt}<|im_end|>"
    f"\n<|im_start|>assistant\nAnswer:")
    return qwen_prompt

def build_llama38b_tf_prompt(prompt):
    llama_prompt=(f"<|begin_of_sentence|>system\nYou are a strict fact-checker. Judge whether the following statement is factually correct based only on your knowledge.\n "
                    f"Respond with exactly one word: either 'True' or 'False'. Do not provide explanations.\n<|end_of_sentence|>"
              f"<|begin_of_sentence|>user\n{prompt}<|end_of_sentence|>"
              f"\n<|begin_of_sentence|>assistant\nAnswer:")

    return llama_prompt

def build_qwen_ab_prompt(question,target_new, target_true,target_random=None, statement=False):
    if target_random is not None:
        prompt = ("<|im_start|>system\nYou are a strict fact-checker. As of 2025-08-29, evaluate the following four statements (A, B, C, D).\n"
                "Decide which single statement is most likely factually correct based on your knowledge. "
                "Respond with exactly one capital letter: A, B, C, or D. Do not explain your choice."
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "Statements:\n A: {}\nB: {}\nC: {}\nD: None of the above is true<|im_end|>\n"
                "<|im_start|>assistant\nAnswer:")
        ori_statements = [target_new, target_true, target_random]
        indexed_statements = list(enumerate(ori_statements))
        random.shuffle(indexed_statements)
        shuffled_statements = [s for (i, s) in indexed_statements]
        ori_index = [i for (i, s) in indexed_statements]
        gt = shuffled_statements.index(target_new)
        A, B, C = shuffled_statements[0],shuffled_statements[1],shuffled_statements[2]
        return [prompt.format(A, B, C)], gt, ori_index
    if statement:
        prompt = ("<|im_start|>system\nThere are two statements labeled A and B. "
                    "As of 2025-08-29, decide which one is more likely to be true based"
                    " on your general world knowledge. Reply with only A or B—no other"
                    " text, no punctuation, and no explanation.<|im_end|>\n"
                    "<|im_start|>user\nStatements:\n A: {}\nB: {}<|im_end|>"
                    "\n<|im_start|>assistant\nAnswer:")
        first_prompt = prompt.format(target_new, target_true)
        second_prompt = prompt.format(target_true, target_new)
    else:
        prompt=("<|im_start|>system\nThere is a question together with two possible answer candidates"
                   " marked with 'A' and 'B' respectively. Based on your knowledge,"
                   " please determine which candidate is the better answer to the "
                   "question. Reply with only A or B.<>\n<|im_end|>\n"
                    "<|im_start|>user\nQuestion: {}\nCandidate A: {}."
                   "\nCandidate B: {}.<|im_end|>\n<|im_start|>assistant\nAnswer:")
        first_prompt=prompt.format(question,target_new, target_true)
        second_prompt=prompt.format(question,target_true, target_new)
    return first_prompt, second_prompt

def build_llama38b_ab_prompt(question,target_new, target_true, target_random=None,statement=False):
    if target_random is not None:
        prompt = ("<|begin_of_sentence|>system\nYou are a strict fact-checker. As of 2025-08-29, evaluate the following four statements (A, B, C, D).\n"
                "Decide which single statement is most likely factually correct based on your knowledge. "
                "Respond with exactly one capital letter: A, B, C, or D. Do not explain your choice."
                "<|end_of_sentence|>\n"
                "<|begin_of_sentence|>user\n"
                "Statements:\n A: {}\nB: {}\nC: {}\nD: None of the above is true<|end_of_sentence|>\n"
                "<|begin_of_sentence|>assistant\nAnswer:")
        ori_statements = [target_new, target_true, target_random]
        indexed_statements = list(enumerate(ori_statements))
        random.shuffle(indexed_statements)
        shuffled_statements = [s for (i, s) in indexed_statements]
        ori_index = [i for (i, s) in indexed_statements]
        gt = shuffled_statements.index(target_new)
        A, B, C = shuffled_statements[0],shuffled_statements[1],shuffled_statements[2]
        return [prompt.format(A, B, C)], gt, ori_index
    if statement:
        prompt = ("<|begin_of_sentence|>system\nThere are two statements labeled A and B. "
                    "As of 2025-08-29, decide which one is more likely to be true based"
                    " on your general world knowledge. Reply with only A or B—no other"
                    " text, no punctuation, and no explanation.<|end_of_sentence|>\n"
                    "<|begin_of_sentence|>user\nStatements:\n A: {}\nB: {}<|end_of_sentence|>"
                    "\n<|begin_of_sentence|>assistant\nAnswer:")
        first_prompt = prompt.format(target_new, target_true)
        second_prompt = prompt.format(target_true, target_new)
    else:
        prompt=("<|begin_of_sentence|>system\nThere is a question together with two possible answer candidates"
                   " marked with 'A' and 'B' respectively. Based on your knowledge,"
                   " please determine which candidate is the better answer to the "
                   "question. Reply with only A or B.<>\n<<|end_of_sentence|>\n"
                    "<|begin_of_sentence|>user\nQuestion: {}\nCandidate A: {}."
                   "\nCandidate B: {}.<|end_of_sentence|>\n<|begin_of_sentence|>assistant\nAnswer:")
        first_prompt=prompt.format(question,target_new, target_true)
        second_prompt=prompt.format(question,target_true, target_new)
    return first_prompt, second_prompt

def test_a_b(cfg,model,tok,ab_prompts,which_correct,key,question=None,statement=False):
    target_new,target_true=ab_prompts
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=build_llama38b_ab_prompt(question,target_new,target_true,statement=statement)
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=build_qwen_ab_prompt(question,target_new,target_true,statement=statement)
    suffix=["_first","_second","third","fourth"]
    keys=[key+suffix[i] for i in range(len(prompts))]
    assert len(prompts)==len(keys)
    n_gen_per_prompt=1#原本用的是10，好像llama3-8b,float32,zsre需要39/40g的样子，所以改成了5
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    # str_lens=[len(prompt) for prompt in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
            top_k=1 if not cfg.topk else 5
    )
    predictions=check_next_word_binary(gen_texts,lens,class1="a",class2="b")
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)

    predicts.extend(list(predictions[:,0]))

    gts=np.repeat(np.array(which_correct).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={}
    for i in range(len(keys)):
        metrics[keys[i]+"_next_acc"]=acc[i].item()
        metrics[keys[i]+"_next_nab_prob"]=np.mean(predictions[i]==-1).item()
    return metrics#如何判断输出是否为True或者False。

def test_abcd(cfg,model,tok,abc_prompts,key,question=None):
    target_new,target_true,target_random=abc_prompts
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts, gt, order_index=build_llama38b_ab_prompt(question,target_new,target_true,target_random)
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts, gt, order_index=build_qwen_ab_prompt(question,target_new,target_true,target_random)
    suffix=["_first","_second","third","fourth"]
    keys=[key+suffix[i] for i in range(len(prompts))]
    assert len(prompts)==len(keys)
    n_gen_per_prompt=1#原本用的是10，好像llama3-8b,float32,zsre需要39/40g的样子，所以改成了5
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    # str_lens=[len(prompt) for prompt in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
            top_k=1 if not cfg.topk else 5
    )
    predictions=check_next_word_binary(gen_texts,lens,class1="a",class2="b",class3="c",class4="d")
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)

    predicts.extend(list(predictions[:,0]))
    abcd_orders.extend(order_index)

    gts=np.repeat(np.array([gt]).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={}
    for i in range(len(keys)):
        metrics[keys[i]+"_next_acc"]=acc[i].item()
        metrics[keys[i]+"_next_nab_prob"]=np.mean(predictions[i]==-1).item()
    return metrics#如何判断输出是否为True或者False。

def test_true_false(cfg,model,tok,prompts,which_correct,key):
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_llama38b_tf_prompt(prefix) for prefix in prompts]
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_qwen_tf_prompt(prefix) for prefix in prompts]
    suffix=["_first","_second","_third","_fourth"]
    keys=[key+suffix[i] for i in range(len(prompts))]
    assert len(prompts)==len(keys)
    n_gen_per_prompt=1#原本用的是10，好像llama3-8b,float32,zsre需要39/40g的样子，所以改成了5
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    # str_lens=[len(prompt) for prompt in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
            top_k=1 if not cfg.topk else 5
    )
    predictions=check_next_word_binary(gen_texts,lens)
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)

    predicts.extend(list(predictions[:,0]))


    gts=np.repeat(np.array(which_correct).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={}
    for i in range(len(keys)):
        metrics[keys[i]+"_next_acc"]=acc[i].item()
        metrics[keys[i]+"_next_ntf_prob"]=np.mean(predictions[i]==-1).item()
    return metrics#如何判断输出是否为True或者False。

def replace_tf_with_detailed_acc(metrics):
    suffixes=["first","second","third","fourth"]
    new_metrics={}
    for key in metrics:
        if key!="rewrite_tf_probs" and key!="rewrite_tf_correct":
            continue
        for i in range(len(suffixes)):
            new_metrics[key+"_"+suffixes[i]]=int(metrics[key][i])
    return new_metrics


def test_true_false_prob(#暂停使用，因为lweval仅仅评估有instruct能力的。可以正确输出true/false。
        cfg,
        model,
        tok,
        prefixes: typing.List[str],
        which_correct: list,
        target_new: str,
        target_true: str,
        key: str,
):
    if "llama" in cfg.llms.name.replace("/","-"):
        prefixes=[build_llama38b_tf_prompt(prefix) for prefix in prefixes]
    elif "qwen" in cfg.llms.name.replace("/","-"):
        prefixes=[build_qwen_tf_prompt(prefix) for prefix in prefixes]
    device = next(model.parameters()).device
    prefix_lens = [len(n) for n in tok(prefixes)["input_ids"]]
    prompt_tok = tok(
        prefixes,
        padding=True,
        return_tensors="pt",
    ).to(device)
    if cfg.llms.name=="meta-llama/Llama-3-8B-Instruct":#这是因为提示里面最后一个词是\n，不需要再输出空格了。
        a_tok, b_tok = (tok(f"{n}")["input_ids"] for n in [target_new, target_true])
    else:#提示里面最后一个词是:需要空格。
        a_tok, b_tok = (tok(f" {n}")["input_ids"] for n in [target_new, target_true])

    if 'llama' in model.config._name_or_path.lower():
        a_tok = a_tok[1:]
        b_tok = b_tok[1:]
        prefix_lens = [lengths - 1 for lengths in prefix_lens]

    with torch.no_grad():
        logits = model(**prompt_tok).logits

    if 'llama' in model.config._name_or_path.lower():
        logits = logits[:, 1:, :]

    #比较一下True和False各自的概率，以及最大的概率是哪一个token即可。
    a_probs=np.zeros((logits.size(0),), dtype=np.float32)
    b_probs = np.zeros((logits.size(0),), dtype=np.float32)
    max_tokens= np.zeros((logits.size(0),), dtype=np.float32)
    for i in range(logits.size(0)):
        probi=torch.nn.functional.log_softmax(
            logits[i, prefix_lens[i] - 1, :], dim=0
        )
        a_probs[i]=probi[a_tok].item()
        b_probs[i]=probi[b_tok].item()
        max_tokens[i]=probi.argmax().item()
    #那么接下来只需要比较即可。根据which correct比较。
    correct_probs=[]
    correct_max=[]
    for i in range(len(which_correct)):
        label=which_correct[i]
        if label==0:#也就是说target new才正确。
            correct_probs.append(a_probs[i]>=b_probs[i])
            correct_max.append(max_tokens[i]==a_tok[0])
        else:#说明target true才正确。
            correct_probs.append(a_probs[i]<b_probs[i])
            correct_max.append(max_tokens[i]==b_tok[0])

    ret={key+"_probs":correct_probs,key+"_correct":correct_max}
    metrics=replace_tf_with_detailed_acc(ret)
    return metrics

def test_inversion_tf(record,cfg,model,tok,):
    prompts = [record["statement_after_inversion"]]
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_llama38b_tf_prompt(prefix) for prefix in prompts]
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_qwen_tf_prompt(prefix) for prefix in prompts]
    n_gen_per_prompt=1#原本用的是10，好像llama3-8b,float32,zsre需要39/40g的样子，所以改成了5
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    # str_lens=[len(prompt) for prompt in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
            top_k=1 if not cfg.topk else 5
    )
    predictions=check_next_word_binary(gen_texts,lens)
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)

    inversion_tf_predicts.extend(gen_texts)


    gts=np.repeat(np.array([0]).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    return {
        "inversion_tf_acc": acc
    }#如何判断输出是否为True或者False。

def get_statements(prompt, subject, target):
    if isinstance(target, list):
        return [prompt.format(subject) + " " + t for t in target]
    return prompt.format(subject) + " " + target

def lw_eval(record,cfg,model,tok):
    metrics={}
    if cfg.data != "counterfact_2000":
        abc_prompts = get_statements(record["prompt"],record["subject"],[record["answer_a"],record["answer_b"],record["target_random"]])

        metrics_abcd = test_abcd(cfg,model,tok,abc_prompts,key="rewrite_abcd")
        metrics=metrics|metrics_abcd
    tf_prompts=record["efficacy_evaluation"][:2]
        # metrics=test_true_false_prob(cfg,model,tok,prob_prompts[0],which_correct[0],
        #                               target_new,target_true,keys[0])
    metrics_tf=test_true_false(cfg,model,tok,tf_prompts,[1,0],key="rewrite_tf")
        # metrics=metrics|metrics_next
    metrics=metrics|metrics_tf
    if cfg.data != "counterfact_2000":
        ab_prompts=[record["answer_a"],record["answer_b"]]
        # if key in record:
        metrics_ab=test_a_b(cfg,model,tok,ab_prompts,which_correct=[0,1],
                    key="rewrite_ab",question=record["question"],statement=False)
        metrics=metrics|metrics_ab
    abs_prompts=[tf_prompts[1],tf_prompts[0]]
    metrics_abs=test_a_b(cfg,model,tok,abs_prompts,which_correct=[0,1],
                 key="rewrite_abs",question=record.get("question", None),statement=True)
    metrics=metrics|metrics_abs

    if cfg.data != "counterfact_2000":
        metrics_random=lw_eval_random(record, cfg, model, tok)
        metrics=metrics|metrics_random
    return metrics

def lw_eval_random(record,cfg,model,tok):
    metrics={}
    eff=record["efficacy_evaluation"][:2]
    tf_prompts=[eff[0],eff[1].replace(record["target_new"],record["target_random"])]
        # metrics=test_true_false_prob(cfg,model,tok,prob_prompts[0],which_correct[0],
        #                               target_new,target_true,keys[0])
    metrics_tf=test_true_false(cfg,model,tok,tf_prompts,[1,0],key="rewrite_rtf")
        # metrics=metrics|metrics_next
    metrics=metrics|metrics_tf
    ab_prompts=[record["target_random"],record["answer_b"]]
    # if key in record:
    metrics_ab=test_a_b(cfg,model,tok,ab_prompts,which_correct=[0,1],
                 key="rewrite_rab",question=record["question"],statement=False)
    metrics=metrics|metrics_ab
    abs_prompts=[tf_prompts[1],tf_prompts[0]]
    metrics_abs=test_a_b(cfg,model,tok,abs_prompts,which_correct=[0,1],
                 key="rewrite_rabs",question=record["question"],statement=True)
    metrics=metrics|metrics_abs
    return metrics

def true_false_probs(record, cfg, model, tok):
    prompts=record["efficacy_evaluation"][:2]
    if "llama" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_llama38b_tf_prompt(prefix) for prefix in prompts]
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        prompts=[build_qwen_tf_prompt(prefix) for prefix in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts, top5_tokens, top5_props =generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=1,
            max_out_len=max_out_len,
            output_top5_tokens=True,
            top_k=1 if not cfg.topk else 5
    )
    probs = torch.zeros((2,2)).to(model.device)

    for i in range(2):
        words = [tok.decode(t) for t in top5_tokens[i]]
        words = [w.lower().strip() for w in words]
        for j in range(len(words)):
            if words[j] == 'true':
                probs[i,0]+=top5_props[i,j]
            elif words[j] == 'false':
                probs[i, 1]+=top5_props[i,j]
    tf_props.append(probs.cpu().numpy())

def test_unlearning_ab(record,cfg,model,tok):
    prompt,target_new,target_true=record["prompt"].format(record["subject"]),record["target_new"], record["target_true"]
    if "llama" in cfg.llms.name.replace("/","-").lower():
        template = ("<|begin_of_sentence|>system\nYou're an AI that completes sentences by choosing the correct option. "
                    "Always output ONLY the letter (A or B). No explanation.<|end_of_sentence|>\n"
                    "<|begin_of_sentence|>user\n{} [A: {} / B: {}]<|end_of_sentence|>"
                    "\n<|begin_of_sentence|>assistant\nAnswer:")
        prompts = [template.format(prompt,target_new,target_true), template.format(prompt,target_true,target_new)]
    elif "qwen" in cfg.llms.name.replace("/","-").lower():
        raise ValueError("Qwen unlearning_ab_test 尚未实现")
    n_gen_per_prompt=1#原本用的是10，好像llama3-8b,float32,zsre需要39/40g的样子，所以改成了5
    lens=np.array([len(prompt.split()) for prompt in prompts])
    lens=np.repeat(lens.reshape(-1,1),n_gen_per_prompt,axis=1).reshape(-1)
    # str_lens=[len(prompt) for prompt in prompts]
    inp_tok = tok(prompts, padding=True, return_tensors="pt")
    max_out_len=inp_tok["input_ids"].shape[1]+1
    gen_texts=generate_fast(
            model,
            tok,
            prompts,
            n_gen_per_prompt=n_gen_per_prompt,
            max_out_len=max_out_len,
            top_k=1 if not cfg.topk else 5
    )
    predictions=check_next_word_binary(gen_texts,lens,class1="a",class2="b")
    predictions=np.array(predictions).reshape(-1,n_gen_per_prompt)

    unleaning_ab_predictions.extend(gen_texts)

    gts=np.repeat(np.array([0,1]).reshape(-1,1),n_gen_per_prompt,axis=1)
    acc=np.mean(predictions==gts,axis=1)
    metrics={
        "unlearning_ab_acc": acc[0],
        "unlearning_ab2_acc": acc[1]
    }
    return metrics#如何判断输出是否为True或者False。
