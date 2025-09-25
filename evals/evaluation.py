from evals.counterfact import eval_counterfact
from evals.zsre import eval_zsre
from evals.wiki_cf import eval_wiki_cf
from evals.mquake_cf import eval_mquake_cf

def eval_one_edit(cfg,model,tok,edit):
    metrics=None
    data_name=cfg.data
    if data_name in ["multi_counterfact_20877","counterfact_2000"]:
        metrics=eval_counterfact(cfg,model,tok,edit)
    elif data_name=="zsre_mend_eval_19086":
        metrics=eval_zsre(cfg,model,tok,edit)
    elif data_name=="mquake_cf_9218":
        metrics=eval_mquake_cf(cfg,model,tok,edit)
    elif data_name=="wiki_cf_2266":
        metrics=eval_wiki_cf(cfg,model,tok,edit)
    elif data_name=="zsre":
        metrics=eval_zsre(cfg,model,tok,edit)
    else:
        raise ValueError("dataset {} not recognized".format(data_name))
    return metrics


