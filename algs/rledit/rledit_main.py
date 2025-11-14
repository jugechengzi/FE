from .datasets.base import make_loader
from .model import make_model
from .rledit import RLEDIT

# multi_counterfact_20877,zsre_mend_eval_19086,wiki_cf_2266,mquake_cf_9218
def apply_rledit_to_model(model, tok, data, cfg):    
    if cfg.data == "multi_counterfact_20877":
        from .datasets import counterfact
        data_class = counterfact.COUNTERFACTDataset
    elif cfg.data == "zsre_mend_eval_19086":
        from .datasets import zsre
        data_class = zsre.ZSREDataset
    else:
        raise NotImplementedError(f"Dataset {cfg.data} not implemented!")

    train_loader, valid_loader = make_loader(cfg, data_class, tok, model.device)

    editor = RLEDIT(cfg, model)
    return editor.run_return_edited_model(train_loader, valid_loader)