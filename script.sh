#multi_counterfact_20877,zsre_mend_eval_19086,wiki_cf_2266,mquake_cf_9218,counterfact_2000
CUDA_VISIBLE_DEVICES=3 nohup python main.py llms=glm4-9b algs=rect data=multi_counterfact_20877 > ./outputs/rect_glm4_m.out &

#test_only=True negetive_prompt_test=True gpu=3 lw_eval=False data=wiki_cf_2266 model_dtype=bfloat16
#正编辑反评测
CUDA_VISIBLE_DEVICES=2 nohup python main.py  algs=alphaedit llms=qwen2.5-7b gpu=0 test_only=True negetive_prompt_test=True save_name="negetive_float32_new" load_name="float32_new" \
 lw_eval=False data=wiki_cf_2266 > nohup.out &

CUDA_VISIBLE_DEVICES=1 nohup python main.py  algs=memit llms=llama3-8b gpu=0 test_only=True negetive_prompt_test=True save_name="negetive_last_k" load_name="last_k" \
 lw_eval=False data=multi_counterfact_20877 > nohup.out &


CUDA_VISIBLE_DEVICES=0 nohup python main.py  algs=memit llms=qwen2.5-7b gpu=0 test_only=True negetive_prompt_test=True save_name="negetive_f32_new_lti" load_name="float32_new_lti" \
 lw_eval=False data=mquake_cf_9218 > nohup.out &

nohup python main.py  algs=rect llms=qwen2.5-7b gpu=0 test_only=True negetive_prompt_test=True save_name="negetive_default" \
 lw_eval=False data=zsre_mend_eval_19086 model_dtype=bfloat16 cache_dir=/home/weiliu/student/xhm/edit-cache \
 > ./outputs/rect-qwen-final_negetive_p_eval_z.out &

#反编辑
CUDA_VISIBLE_DEVICES=3 nohup python main.py algs=prune llms=llama3-8b gpu=0 seed=1 negetive_prompt_test=True save_name="bf16_seed1_negetive_edit" \
 data=multi_counterfact_20877 > nohup.out &

#反编辑正评测
nohup python main.py algs=memit llms=qwen2.5-7b gpu=0 test_only=True save_name="lti_negetive_edit_new" load_name="lti_negetive_edit_new" \
 lw_eval=False data=multi_counterfact_20877 > nohup.out &

nohup python main.py algs=prune llms=qwen2.5-7b gpu=3 test_only=True save_name="negetive_edit_new" load_name="negetive_edit_new" \
 lw_eval=False data=mquake_cf_9218 > nohup.out &
#反编辑反评测
nohup python main.py algs=memit llms=qwen2.5-7b gpu=0 test_only=True negetive_prompt_test=True save_name="lti_negetive_edit_new_test" load_name="lti_negetive_edit_new" \
 lw_eval=False data=multi_counterfact_20877 > nohup.out &

nohup python main.py algs=prune llms=qwen2.5-7b gpu=2 test_only=True negetive_prompt_test=True save_name="negetive_edit_new_test" load_name="negetive_edit_new" \
 lw_eval=False data=mquake_cf_9218 > nohup.out &

#正编辑
CUDA_VISIBLE_DEVICES=0 nohup python main.py algs=memit llms=qwen2.5-7b gpu=0 data=multi_counterfact_20877 model_dtype=float32 lti=True save_name="float32_new_lti" > ./outputs/nohup.out &
CUDA_VISIBLE_DEVICES=0 nohup python main.py algs=prune llms=llama3-8b gpu=0 seed=1 data=multi_counterfact_20877 save_name="seed1" > ./outputs/nohup.out &

#正测评
CUDA_VISIBLE_DEVICES=2 nohup python main.py algs=memit llms=llama3-8b gpu=0 data=multi_counterfact_20877 test_only=True lw_eval=False load_name="last_k" save_name="last_k_default" > nohup.out &
#multi_counterfact_20877,zsre_mend_eval_19086,wiki_cf_2266,mquake_cf_9218,counterfact_2000