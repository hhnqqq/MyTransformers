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
from common.lora_modules.salora import LinearWithSALoRA
from common.lora_modules.mola import LinearWithMoLA
from common.lora_modules.nlora import LinearWithNLoRA
from common.lora_modules.nora import LinearWithNoRA
from common.lora_modules.randlora import LinearWithRandLoRA
from common.lora_modules.dude import LinearWithDude
from common.lora_modules.lora_ga_pro import LinearWithLoRAGAPro

lora_variants = {
    "use_dora": (LinearWithDoRA, lambda a: {}, ""),
    "use_mos_lora": (LinearWithMosLoRA, 
                    lambda a: {"weight_ab_mixer_init_method": a.weight_ab_mixer_init_method}, ""),
    "use_me_lora": (LinearWithMELoRA, 
                    lambda a: {"me_lora_n_split": a.me_lora_n_split, 
                                "forward_method": a.me_lora_forward_method}, ""),
    "use_lora_ga": (LinearWithLoRAGA, 
                    lambda a: {}, 
                    lambda a: f". The initialization of LoRA-GA requires some time which depends on args.lora_ga_n_steps: {a.lora_ga_n_steps}"),
    "use_rslora": (LinearWithRSLoRA, lambda a: {}, ""),
    "use_pissa": (LinearWithPiSSA, 
                    lambda a: {"fast_svd_n_iters": a.pissa_n_iters, 
                            "keep_init_weights": a.pissa_keep_init_weights}, 
                    ". The initialization of Pissa requires some time especially for full svd decomposition, waiting..."),
    "use_olora": (LinearWithOLoRA, 
                    lambda a: {}, 
                    ". The initialization of Olora requires some time, waiting..."),
    "use_vera": (LinearWithVeRA, lambda a: {"lambda_b_init_method":a.lambda_b_init_method, "lambda_d_init_method":a.lambda_d_init_method,}, ""),
    "use_adalora": (LinearWithAdaLoRA, lambda a: {"init_r": a.init_r}, ""),
    "use_delta_lora": (LinearWithDeltaLoRA, 
                        lambda a: {"update_ratio": a.delta_lora_update_ratio}, ""),
    "use_lora_moe": (LinearWithLoRAMoE, 
                    lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, 
                                "lora_moe_top_k": a.lora_moe_top_k}, ""),
    "use_milora": (LinearWithMILoRA, 
                    lambda a: {"fast_svd_n_iters": a.milora_n_iters}, 
                    ". The initialization of milora requires some time especially for full svd decomposition, waiting..."),
    "use_plora": (LinearWithPLoRA, 
                    lambda a: {"plora_momentum": a.plora_momentum}, 
                    lambda a: f". PLoRA will reset lora weights with momentum: {a.plora_momentum} at every step."),
    "use_mora": (LinearWithMoRA, lambda a: {"mora_type": a.mora_type}, ""),
    "use_gora": (LinearWithGoRA, 
                lambda a: {"gora_init_method": a.gora_init_method,
                            "gora_rank_stablize": a.gora_rank_stablize,
                            "gora_dynamic_scaling": a.gora_dynamic_scaling}, ""),
    "use_increlora": (LinearWithIncreLoRA, lambda a: {"init_r": a.init_r}, ""),
    "use_salora": (LinearWithSALoRA, 
                    lambda a: {"init_r": a.init_r, "target_r": a.target_r}, ""),
    "use_mola": (LinearWithMoLA,
                lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, 
                            "lora_moe_top_k": a.lora_moe_top_k}, ""),
    "use_nlora": (LinearWithNLoRA,
                lambda a: {"weight_ab_mixer_init_method": None}, ""),
    "use_nora":  (LinearWithNoRA,
                lambda a: {"fast_svd_n_iters": a.nora_n_iters}, ""),
    "use_randlora": (LinearWithRandLoRA, 
                     lambda a: {"lambda_b_init_method":a.lambda_b_init_method, "lambda_d_init_method":a.lambda_d_init_method,}, ""),
    "use_dude": (LinearWithDude,
                 lambda a: {"fast_svd_n_iters":a.pissa_n_iters}, ""),
    "use_loraga_pro": (LinearWithLoRAGAPro,
                       lambda a: {"rank_stablize":a.lora_ga_pro_rank_stablize,
                                  "dynamic_scaling":a.lora_ga_pro_dynamic_scaling}, "")
}

def get_lora_layer_class(args):
    lora_layer_class = LinearWithLoRA
    variant_config = {}
    variant_print = ""
    
    if getattr(args, "relora_steps", False) or getattr(args, "relora_counts", False):
        variant_print = f". Will reset lora weights every {args.relora_steps} global update steps."
    else:
        for attr_name, (cls, config_fn, print_msg) in lora_variants.items():
            if getattr(args, attr_name, False):
                lora_layer_class = cls
                variant_config = config_fn(args)
                variant_print = print_msg(args) if callable(print_msg) else print_msg
                break
    
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

def check_applied_lora(model):
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

        if not check_applied_lora(model):
            print_rank_0(f'--->Can not find replace modules: {args.replace_modules} in model, LoRA is targeting all-linear now.')
            args.replace_modules = ['all-linear']
            switch_to_lora(model, args)

        if args.lora_fa:
            lora_weight = ['weight_b']
        elif args.use_vera or args.use_randlora:
            lora_weight = ['lambda']
        else:
            lora_weight = ['weight_a','weight_b']
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