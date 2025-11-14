from typing import Union, Tuple, List, Dict
from omegaconf import DictConfig

import math
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

DATASET_MAP = {
    "multi_counterfact_20877": {
        "train_path": "/home/weiliu/student/xhm/data/multi_counterfact_eval.json",
        "valid_path": "/home/weiliu/student/xhm/data/multi_counterfact_20877.json",
        "negetive_prompt_path": "/home/weiliu/student/xhm/data/mcf_negetive_prompt.json"
    },
    "zsre_mend_eval_19086": {
        "train_path": "/home/weiliu/student/xhm/data/zsre_train.json",
        "valid_path": "/home/weiliu/student/xhm/data/zsre_mend_eval_19086.json",
        "negetive_prompt_path": "/home/weiliu/student/xhm/data/zsre_negetive_prompt.json"
    }
}

class BaseDataset(Dataset):

    def __init__(
        self,
        config: DictConfig,
        path: str,
        tok: AutoTokenizer,
        device: Union[int, str, torch.device],
        negetive_prompt_test: bool = False
    ):
        self.config = config
        with open(path) as file:
            self.data = json.load(file)
        self.tok = tok
        self.device = device
        self.negetive_prompt_test = negetive_prompt_test


    def __len__(self):
        return len(self.data)


    def collate_fn(
        self,
        tuples: Tuple[Dict[str, Dict[str, torch.LongTensor]]]
    ) -> Dict[str, List[Dict[str, torch.LongTensor]]]:
        tuples: Dict[str, List[Dict[str, torch.LongTensor]]] = {
            k: sorted(
                [t[k] for t in tuples],
                key = lambda x: x["attention_mask"].sum().item(),
                reverse = True
            )
            for k in tuples[0].keys()
        }
        
        return {
            k: [
                self.pad_tok_tuples(v[n_batch * self.config.dataset_batch_size:(n_batch + 1) * self.config.dataset_batch_size])
                for n_batch in range(math.ceil(self.config.n_edits / self.config.dataset_batch_size))
            ]
            for k, v in tuples.items()
        }
        

    def pad_tok_tuples(
        self,
        tok_tuples: List[Dict[str, torch.LongTensor]]
    ) -> Dict[str, torch.LongTensor]:
        
        return {
            k: pad_sequence(
                [t[k].squeeze(0) for t in tok_tuples],
                batch_first = True,
                padding_value = -100 if k == "labels" else 0
            ).to(self.device)
            for k in tok_tuples[0].keys()
        }



def make_loader(
    config: DictConfig,
    data_class,
    tok,
    device
) -> Tuple[DataLoader]:
    dataset_info = DATASET_MAP[config.data]
    train_path = dataset_info["train_path"]
    valid_path = dataset_info["valid_path"]
    negetive_prompt_path = dataset_info["negetive_prompt_path"]

    train_set = data_class(
        config.algs,
        train_path,
        tok,
        device
    )


    if config.negetive_prompt_test:
        valid_set = data_class(
            config.algs,
            negetive_prompt_path,
            tok,
            device,
            negetive_prompt_test=True
        )
    else:
        valid_set = data_class(
            config.algs,
            valid_path,
            tok,
            device
        )

    train_loader = DataLoader(
        train_set,
        config.algs.n_edits,
        True,
        collate_fn = train_set.collate_fn,
        drop_last = True
    )


    valid_loader = DataLoader(
        valid_set,
        config.algs.n_edits,
        True,
        collate_fn = valid_set.collate_fn,
        drop_last = True
    )


    return train_loader, valid_loader