from typing import Dict
import torch

from .base import BaseDataset


class ZSREDataset(BaseDataset):
    
    def __getitem__(self, idx) -> Dict[str, Dict[str, torch.LongTensor]]:
        row = self.data[idx]
        if row.get("loc", None) is None:
            prompt = row["prompt"].format(row["subject"])
            if self.negetive_prompt_test:
                prompt = row["negetive_prompt"]
            equiv_prompt = row["paraphrase_prompts"][0]
            answer = row["target_new"]
            unrel_prompt = row["neighborhood_prompts"][0] + "?"
            unrel_answer = row["neighborhood_prompts_answers"][0]
        else:
            prompt = row["src"]
            equiv_prompt = row["rephrase"]
            answer = row["ans"]
            unrel_prompt = row["loc"] + "?"
            unrel_answer = row["loc_ans"]
    
        return {
            "edit_tuples": self.tok_tuples(prompt, answer),
            "equiv_tuples": self.tok_tuples(equiv_prompt, answer),
            "unrel_tuples": self.tok_tuples(unrel_prompt, unrel_answer)
        }
        

    def tok_tuples(
        self,
        prompt: str,
        answer: str
    ) -> Dict[str, torch.LongTensor]:

        answer = " " + answer
        tok_prompt = self.tok(
            prompt,
            return_tensors="pt",
        )
        tok_answer = self.tok(
            answer,
            return_tensors="pt",
            add_special_tokens=False
        )

        tok_tuples = {
            key: torch.cat((value, tok_answer[key][:, :-1]), -1)
            for key, value in tok_prompt.items()
        }
        
        tok_tuples["labels"] = torch.cat((
            torch.full(tok_prompt["input_ids"].shape, -100)[:, 1:],
            tok_answer["input_ids"]
        ), -1)

        return tok_tuples