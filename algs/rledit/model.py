from omegaconf import DictConfig
from transformers import AutoModelForCausalLM

from .util import get_module


def make_model(config: DictConfig, model: AutoModelForCausalLM) -> AutoModelForCausalLM:
    for param in model.parameters():
        param.requires_grad = False
        
    for module_name in config.llms.edit_modules:
        module = get_module(model, module_name)
        module.weight.requires_grad = True
        
    return model