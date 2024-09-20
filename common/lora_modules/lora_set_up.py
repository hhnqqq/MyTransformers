from argparse import Namespace
from typing import Union, List

from common.utils import print_rank_0
from common.lora_modules.lora import *
from common.lora_modules.dora import LinearWithDoRA
from common.lora_modules.melora import LinearWithMELoRA
from common.lora_modules.plora import LinearWithPLoRA
from common.lora_modules.lora_ga import LinearWithLoRAGA
from common.lora_modules.mos_lora import LinearWithMosLoRA

def get_lora_layer_class(args):
    variant_config = {}
    if args.use_dora:
        lora_layer_class = LinearWithDoRA
    elif args.plora_steps:
        lora_layer_class = LinearWithPLoRA
        variant_config = dict(plora_steps=args.plora_steps)
    elif args.use_mos_lora:
        lora_layer_class = LinearWithMosLoRA
        variant_config = dict(weight_ab_mixer_init_method=args.weight_ab_mixer_init_method)
    elif args.use_me_lora:
        lora_layer_class = LinearWithMELoRA
        variant_config = dict(me_lora_n_split=args.me_lora_n_split)
    elif args.use_lora_ga:
        lora_layer_class = LinearWithLoRAGA
    else:
        lora_layer_class = LinearWithLoRA

    print_rank_0(f'Using lora variant: {lora_layer_class.__name__}', rank=args.global_rank)
    return lora_layer_class, variant_config

def switch_to_lora(model: nn.Module, 
                   args: Namespace,
                   replace_names: Optional[Union[str, List[str]]] = None, 
                   rank: int = 4, 
                   lora_scaler: int = 32, 
                   lora_dropout: Optional[float] = None,
                   transposition: bool = False):
    """
    Switch function for lora, responsible for replacing Linear layer with LinearWithLoRA layer

    Args:
        model: Any pytorch model.
        replace_names: List of module names to be replaced by LoRA.
        rank: Rank for LoRA.
        lora_scaler: Scaler for LoRA.
        transposition: nn.Linear(x, w) compute xw^T, so the weight should in shape [out_feature, in_feature]. Otherwise, transposition should be set to True
        use_dora: Weather to use dora
        plora_steps: The steps to merge and reset lora weight.
    """
    assert replace_names is not None, 'Replace names can not be None'
    lora_layer_class, variant_config = get_lora_layer_class(args)
    for name, module in model.named_modules():
        replace_tag = False
        for replace_name in replace_names:
            if replace_name in name:
                # Create LoRA layer instance.
                replace_tag = True
                if isinstance(module, LinearWithLoRA):
                    module.merge_and_reset(new_rank=rank)
                elif isinstance(module, nn.Module):
                    if  all(hasattr(module, attr) for attr in ["in_features", "out_features", "weight"]):
                        quant = getattr(module, "quant", False)
                        lora_config = dict(lora_rank=rank, 
                                        lora_scaler=lora_scaler, 
                                        lora_dropout=lora_dropout,
                                        in_features=module.in_features, 
                                        out_features=module.out_features, 
                                        quant=quant)
                        lora_layer = lora_layer_class(**lora_config, **variant_config)
                        # Copy the original weight to the LoRA layer.
                        if transposition:
                            lora_layer.weight = nn.Parameter(module.weight.data.T)
                        else:
                            lora_layer.weight.data = module.weight.data
                        if quant:
                            lora_layer.weight_scaler = module.weight_scaler
                        # Replace the original layer with the LoRA layer.
                        parent = get_parent_model(model, module)
                        setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], lora_layer)
        if not replace_tag and isinstance(module, LinearWithLoRA):
            # Merge weight to avoid unnecessary computing.
            module.merge_and_del()

def setup_lora(model, args, model_config=None):
    if args.use_lora:
        if args.replace_modules is None:
            args.replace_modules = model_config.lora_layers
        switch_to_lora(model, 
                       args,
                       args.replace_modules, 
                       rank=args.lora_rank, 
                       lora_scaler=args.lora_scaler,
                       lora_dropout=args.lora_dropout)
        if args.lora_fa:
            lora_weight = ['weight_b', 'weight_ab_mixer']
        else:
            lora_weight = ['weight_a','weight_b', 'weight_ab_mixer']
        args.enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
        if hasattr(args, 'device'):
            if args.fp16:
                model.to(args.device).half()
            elif args.bf16:
                model.to(args.device).bfloat16()

def recover_linear(model: nn.Module):
    """
    Recover function for lora, responsible for recover LinearWithLoRA layer to Linear layer.

    Args:
        model: Any pytorch model.
    """
    for module in model.modules:
        if isinstance(module, LinearWithLoRA):
            module.merge_and_del()
            linear_layer = nn.Linear(in_features=module.in_features,
                                     out_features=module.out_features,
                                     bias=False,
                                     dtype=module.weight.dtype,
                                     device=module.weight.dtype)
            linear_layer.weight.data = module.weight.data
            parent = get_parent_model(model, module)
            setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], linear_layer)
            


def get_parent_model(parent_model, module):
    """
    Find the parent module for the input module recursively.

    Args:
        parent_model: Root model for the search.
        module: Submodule to find the parent module for.

    Returns:
        Parent module if found, None otherwise.
    """
    for _, sub_module in parent_model._modules.items():
        # Direct sub modules of parent model.
        if sub_module is module:
            return parent_model
        parent = get_parent_model(sub_module, module)
        if parent:
            return parent
    return None