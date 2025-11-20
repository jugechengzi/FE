from evals.counterfact import eval_counterfact
from evals.zsre import eval_zsre
from evals.wiki_cf import eval_wiki_cf
from evals.mquake_cf import eval_mquake_cf

def eval_one_edit(cfg,model,tok,edit):
    metrics=None
    data_name=cfg.data
    if "counterfact" in data_name:
        metrics=eval_counterfact(cfg,model,tok,edit)
    elif "zsre_mend_eval" in data_name:
        metrics=eval_zsre(cfg,model,tok,edit)
    elif "mquake_cf" in data_name:
        metrics=eval_mquake_cf(cfg,model,tok,edit)
    elif "wiki_cf" in data_name:
        metrics=eval_wiki_cf(cfg,model,tok,edit)
    else:
        raise ValueError("dataset {} not recognized".format(data_name))
    return metrics


