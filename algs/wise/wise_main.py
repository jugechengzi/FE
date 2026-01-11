from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from omegaconf import DictConfig
from .WISE import WISE
from .utils import tokenize, get_context_templates, get_load_model_path
from tqdm import tqdm
def apply_wise_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        cfg: DictConfig,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    tok.padding_side = 'left'
    device = f'cuda:{cfg.gpu}'
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=device)
    editor = WISE(model=model, config=cfg.llms, device=device)
    print(f"Executing WISE algorithm for the update: ")
    for request in requests[:10]:
        print(
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']}]"
        )
    for request in tqdm(requests, total=len(requests)):
        request["prompt"]=request["prompt"].format(request["subject"])
        tokens, act_mask, deact_mask = tokenize([request], tokenizer=tok, device=device, context_templates=context_templates, hparams=cfg.llms)
        editor.edit(config=cfg.llms, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    # weights_copy = editor.reset_layer

    return editor



