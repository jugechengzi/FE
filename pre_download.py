#需要安装transformers库,pip install即可。
from transformers import AutoModelForCausalLM, AutoTokenizer

### download wikipedia dataset for K0K0T。
ds_name="wikipedia"
from datasets import load_dataset#pip install datasets==2.18.0
raw_ds = load_dataset(
        ds_name,
    dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name]
    )

llms=["gpt2-xl","EleutherAI/gpt-j-6B"]
for llm in llms:
    model = AutoModelForCausalLM.from_pretrained(
        llm,
        torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(llm)

from modelscope import snapshot_download
llamas=["meta-llama/Llama-3-8B-Instruct","meta-llama/Llama-3.1-8B-Instruct",
              "meta-llama/Llama-3.2-3B-Instruct"]
#llama models are restricted to access, we use modelscope to download.
llamas_modelscope=["LLM-Research/Meta-Llama-3-8B-Instruct","LLM-Research/Meta-Llama-3.1-8B-Instruct",
                   "LLM-Research/Llama-3.2-3B-Instruct"]
for llama in llamas_modelscope:
    model_id=snapshot_download(llama,
                               cache_dir="/home/liubingqing/.cache/modelscope")#将下载的大模型存放在cache_dir目录下。
#这里使用~代表home目录会失败，不知道为什么，会选择创建一个~目录。

#假设我们有2个知识要编辑。
edited_facts=[{"question":"The mother tongue of Danielle Darrieux is","answer":"American English"},
              {"question":"Danielle Darrieux was born in","answer":"China"}]
#现在我有如下2个问题
questions=["What is the mother tongue of Danielle Darrieux?",
           "The mother tongue of Danielle Darrieux is"]
'''
你需要使用那篇论文的代码完成对上述2个问题的回答，写两个问题，是希望你的代码是批处理的，而不是for循环先回答第一个question，然后第二个。
我需要如下5个东西：
1.answers#有两个question，那么答案也应该有2个。
2.ori_dist,ini_dist,enc_dist,dec_dist#论文图4的那4个分布，另外，对于给定的那2个question，按理说都应该回答American English，
也就是说生成2个单词，那么ori_dist应该有两个分布，即ori_dist的shape是[2,vocab_size]，其他3个dist同理。
你需要写一个函数，返回上述5个东西。
def func(edited_facts,questions):
    *****************
    return answers,(ori_dist,ini_dist,enc_dist,dec_dist)
'''

# def zsre():
#     # 需要现有这个启动一下。
#     answer = ""  # 对于一个给定的句子提问，要求输出一个问句和该问句对应的答案，答案必须是原来句子的最后一个名词。例如，
#     Input = "\n" + "对于一个给定的问句，要求输出一个陈述句。例如，给定问句：Where is the place of death of Leo Arons? 输出为陈述句：The place of death of Leo Arons is.  再例如给定疑问句：Which city did Francis Palmer Smith work? 输出为陈述句Francis Palmer Smith worked in the city of.   再例如给定句子，Which company is Michaela Pereira employed by? 输出为陈述句：Michaela Pereira is employed by. 那么给定疑问句：What is the name of the field of work of S. L. Peshtich? 其对应的陈述句输出是什么\n "
#     question = checklen(getText("user", Input))
#     # print("星火:", end="")
#     main(appid, api_key, api_secret, Spark_url, domain, question)
#     z = getText("assistant", answer)
#     text = text[:2]
#
#     outputs = []
#     for i in range(len(data)):
#         sentence = data[i]["prompt"].format(data[i]["subject"])
#         answer = ""
#         # Input = "\n" + "我:给定句子：Apple A5 was created by Apple. 应该输出什么，不要输出思维链，直接返回输出\n "
#         Input = "\n" + "对于一个给定的疑问句，要求输出一个陈述句。参考对话最开始的那个例子，给定疑问句：{} 输出陈述句：\n".format(
#             sentence)
#         question = checklen(getText("user", Input))
#         # print("星火:", end="")
#         main(appid, api_key, api_secret, Spark_url, domain, question)
#         z = getText("assistant", answer)
#         text = text[:6]
#         outputs.append(z[-1]["content"])









