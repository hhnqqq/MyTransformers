from common.lora_modules.lora import *

def switch_to_lora(model: nn.Module, 
                   replace_names: Optional[Union[str, List[str]]] = None, 
                   rank: int = 4, 
                   lora_scaler: int = 32, 
                   lora_dropout: Optional[float] = None,
                   transposition: bool = False, 
                   use_dora: bool = False, 
                   use_mos_lora: bool = False,
                   plora_steps: Optional[int] = None):
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
                        lora_layer = LinearWithLoRA(lora_rank=rank, 
                                                    lora_scaler=lora_scaler, 
                                                    lora_dropout=lora_dropout,
                                                    in_features=module.in_features, 
                                                    out_features=module.out_features, 
                                                    use_dora=use_dora, 
                                                    use_mos_lora=use_mos_lora,
                                                    quant=quant,
                                                    plora_steps=plora_steps)
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

def setup_lora(model, args, model_config):
    if args.use_lora:
        if args.replace_modules is None:
            args.replace_modules = model_config.lora_layers
        switch_to_lora(model, 
                       args.replace_modules, 
                       rank=args.lora_rank, 
                       use_dora=args.use_dora,
                       use_mos_lora=args.use_mos_lora,
                       lora_dropout=args.lora_dropout,
                       plora_steps=args.plora_steps)
        if args.lora_fa:
            lora_weight = ['weight_b', 'weight_ab_mixer']
        else:
            lora_weight = ['weight_a','weight_b', 'weight_ab_mixer']
        args.enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
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
            linear.weight.data = module.weight.data
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