from argparse import Namespace

from common.utils import print_rank_0
from common.lora_modules.lora import *
from common.lora_modules.dora import LinearWithDoRA
from common.lora_modules.melora import LinearWithMELoRA
from common.lora_modules.lora_ga import LinearWithLoRAGA
from common.lora_modules.mos_lora import LinearWithMosLoRA
from common.lora_modules.rslora import LinearWithRSLoRA
from common.lora_modules.pissa import LinearWithPiSSA
from common.lora_modules.olora import LinearWithOLoRA
from common.lora_modules.vera import LinearWithVeRA
from common.lora_modules.lora_moe import LinearWithLoRAMoE
from common.lora_modules.milora import LinearWithMILoRA
from common.lora_modules.delta_lora import LinearWithDeltaLoRA
from common.lora_modules.adalora import LinearWithAdaLoRA
from common.lora_modules.plora import LinearWithPLoRA
from common.lora_modules.mora import LinearWithMoRA
from common.lora_modules.gora import LinearWithGoRA
from common.lora_modules.increlora import LinearWithIncreLoRA

def get_lora_layer_class(args):
    variant_config = dict()
    variant_print = ""
    lora_layer_class = LinearWithLoRA
    if getattr(args, "use_dora", False):
        lora_layer_class = LinearWithDoRA
    elif getattr(args, "use_mos_lora", False):
        lora_layer_class = LinearWithMosLoRA
        variant_config = dict(weight_ab_mixer_init_method=args.weight_ab_mixer_init_method)
    elif getattr(args, "use_me_lora", False):
        lora_layer_class = LinearWithMELoRA
        variant_config = dict(me_lora_n_split=args.me_lora_n_split)
    elif getattr(args, "use_lora_ga", False):
        lora_layer_class = LinearWithLoRAGA
        variant_print = f". The initialization of LoRA-GA requires some time which depends on args.lora_ga_n_steps: {args.lora_ga_n_steps}"
    elif getattr(args, "use_rslora", False):
        lora_layer_class = LinearWithRSLoRA
    elif getattr(args, "use_pissa", False):
        lora_layer_class = LinearWithPiSSA
        variant_config = dict(fast_svd_n_iters=args.pissa_n_iters,
                              keep_init_weights=args.pissa_keep_init_weights)
        variant_print = ". The initialization of Pissa requires some time especially for full svd decomposition, waiting..."
    elif getattr(args, "use_olora", False):
        lora_layer_class = LinearWithOLoRA
        variant_print = ". The initialization of Olora requires some time, waiting..."
    elif getattr(args, 'use_vera', False):
        lora_layer_class = LinearWithVeRA
    elif getattr(args, 'use_adalora', False):
        lora_layer_class = LinearWithAdaLoRA
        variant_config = dict(init_r=args.init_r)
    elif getattr(args, 'use_delta_lora', False):
        lora_layer_class = LinearWithDeltaLoRA
        variant_config = dict(update_ratio=args.delta_lora_update_ratio)
    elif getattr(args, 'use_lora_moe', False):
        lora_layer_class = LinearWithLoRAMoE
        variant_config = dict(lora_moe_n_experts=args.lora_moe_n_experts,
                              lora_moe_top_k=args.lora_moe_top_k)
    elif getattr(args, 'use_milora', False):
        lora_layer_class = LinearWithMILoRA
        variant_config = dict(fast_svd_n_iters=args.milora_n_iters)
        variant_print = ". The initialization of milora requires some time especially for full svd decomposition, waiting..."
    elif getattr(args, 'use_plora', False):
        lora_layer_class = LinearWithPLoRA
        variant_config = dict(plora_momentum=args.plora_momentum)
        variant_print = f". PLoRA will reset lora weights with momentum: {args.plora_momentum} at every step."
    elif getattr(args, 'use_mora', False):
        lora_layer_class = LinearWithMoRA
        variant_config = dict(mora_type=args.mora_type)
    elif getattr(args, 'use_gora', False):
        lora_layer_class = LinearWithGoRA
        variant_config = dict(gora_init_method=args.gora_init_method,
                              gora_rank_stablize=args.gora_rank_stablize,
                              gora_dynamic_scaling=args.gora_dynamic_scaling)
    elif getattr(args, "relora_steps", False) or getattr(args, "relora_counts", False):
        # if args.relora_counts:
        #     args.relora_steps = args.num_global_update_steps // (args.relora_counts + 1)
        variant_print = f". Will reset lora weights every {args.relora_steps} global update steps."
    elif getattr(args, 'use_increlora', False):
        lora_layer_class = LinearWithIncreLoRA
        variant_config = dict(init_r=args.init_r)
    print_rank_0(f'--->Using lora variant: {lora_layer_class.__name__}{variant_print}', rank=args.global_rank)
    return lora_layer_class, variant_config

def switch_to_lora(model: nn.Module, 
                   args: Namespace,
                   transposition: bool = False):
    """
    Switch function for lora, responsible for replacing Linear layer with LinearWithLoRA layer

    Args:
        model: Any pytorch model.
        replace_modules: List of module names to be replaced by LoRA.
        rank: Rank for LoRA.
        lora_scaler: Scaler for LoRA.
        transposition: nn.Linear(x, w) compute xw^T, so the weight should in shape [out_feature, in_feature]. Otherwise, transposition should be set to True
        use_dora: Weather to use dora
        plora_steps: The steps to merge and reset lora weight.
    """
    assert args.replace_modules is not None, 'Replace modules can not be None'
    lora_layer_class, variant_config = get_lora_layer_class(args)
    if args.run_lora_in_fp32:
        print_rank_0('--->Will keep lora weights in float32', args.global_rank)
    for name, module in model.named_modules():
        replace_tag = False
        for module_name in args.replace_modules:
            if module_name in name or (module_name == 'all-linear' and 'lm_head' not in name):
                # Create LoRA layer instance.
                replace_tag = True
                if isinstance(module, LinearWithLoRA):
                    module.merge_and_reset(new_rank=args.rank)
                elif isinstance(module, nn.Module):
                    if  all(hasattr(module, attr) for attr in ["in_features", "out_features", "weight"]):
                        quant = getattr(module, "quant", False)
                        bias = getattr(module, "bias", None)
                        lora_config = LoRAConfig(lora_rank=args.lora_rank, 
                                            lora_scaler=args.lora_scaler, 
                                            lora_dropout=args.lora_dropout,
                                            run_lora_in_fp32=args.run_lora_in_fp32,
                                            weight_a_init_method=args.weight_a_init_method,
                                            weight_b_init_method=args.weight_b_init_method,
                                            in_features=module.in_features, 
                                            out_features=module.out_features,
                                            bias=(bias is not None), 
                                            quant=quant)
                        lora_layer: LinearWithLoRA = lora_layer_class(lora_config, **variant_config)
                        # Copy the original weight to the LoRA layer.
                        if transposition:
                            lora_layer.weight = nn.Parameter(module.weight.data.T)
                        else:
                            lora_layer.weight.data = module.weight.data
                        if quant:
                            lora_layer.weight_scaler = module.weight_scaler
                        lora_layer.bias = bias
                        # Manually init lora weights after read pretrianed weight.
                        lora_layer.init_lora_weights()
                        # Replace the original layer with the LoRA layer.
                        parent = get_parent_model(model, module)
                        setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], lora_layer)
        if not replace_tag and isinstance(module, LinearWithLoRA):
            # Merge weight to avoid unnecessary computing.
            module.merge_and_del()

def check_applyed_lora(model):
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            return True
    return False

def setup_lora(model, args, model_config=None):
    """
    Set up LoRA layers according to `args.replace_modules` or `model_config.lora_layers`

    Args:
        model: Any PyTorch model.
        args: Any data structure that containing LoRA information.
        model_config: Config of the model.
    """
    if args.use_lora:
        if args.replace_modules is not None:
            if isinstance(args.replace_modules, str):
                args.replace_modules = args.replace_modules.split('_')
        else:
            args.replace_modules = getattr(model_config, "lora_layers", None)
        if args.replace_modules:
            print_rank_0(f'--->LoRA targeting modules: {args.replace_modules}', args.global_rank)
        else:
            print_rank_0('--->The replace modules is not provided, LoRA is targating all linear modules.', args.global_rank)
        switch_to_lora(model, args)

        if not check_applyed_lora(model):
            print_rank_0(f'--->Can not find replace modules: {args.replace_modules} in model, LoRA is targeting all-linear now.')
            args.replace_modules = ['all-linear']
            switch_to_lora(model, args)

        if args.lora_fa:
            lora_weight = ['weight_b', 'weight_ab_mixer']
        elif args.use_vera:
            lora_weight = ['lambda']
        else:
            lora_weight = ['weight_a','weight_b', 'weight_ab_mixer']
        args.enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
    model.to(args.device)

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