import torch.nn as nn
from typing import Type, Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
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

@dataclass
class LoRAVariant:
    """
    Configuration class for LoRA variants.
    
    Attributes:
        class_type: The class type of the LoRA variant
        config_generator: Function to generate configuration for the variant
        init_message: Message or function to generate message during initialization
    """
    class_type: Type
    config_generator: Callable
    init_message: Union[str, Callable]

LORA_VARIANTS: Dict[str, LoRAVariant] = {
    "use_dora": LoRAVariant(
                LinearWithDoRA, 
                lambda a: {}, 
                ""),
    "use_mos_lora": LoRAVariant(
                LinearWithMosLoRA, 
                lambda a: {"weight_ab_mixer_init_method": a.weight_ab_mixer_init_method}, 
                ""),
    "use_me_lora": LoRAVariant(
                LinearWithMELoRA, 
                lambda a: {"me_lora_n_split": a.me_lora_n_split, "forward_method": a.me_lora_forward_method}, 
                ""),
    "use_lora_ga": LoRAVariant(
                LinearWithLoRAGA, 
                lambda a: {}, 
                lambda a: f". The initialization of LoRA-GA requires some time which depends on args.lora_ga_n_steps: {a.lora_ga_n_steps}"),
    "use_rslora": LoRAVariant(LinearWithRSLoRA, lambda a: {}, ""),
    "use_pissa": LoRAVariant(
                LinearWithPiSSA, 
                lambda a: {"fast_svd_n_iters": a.pissa_n_iters, "keep_init_weights": a.pissa_keep_init_weights}, 
                ". The initialization of Pissa requires some time especially for full svd decomposition, waiting..."),
    "use_olora": LoRAVariant(
                LinearWithOLoRA, 
                lambda a: {}, 
                ". The initialization of Olora requires some time, waiting..."),
    "use_vera": LoRAVariant(
                LinearWithVeRA, 
                lambda a: {"lambda_b_init_method":a.lambda_b_init_method, "lambda_d_init_method":a.lambda_d_init_method,}, 
                ""),
    "use_adalora": LoRAVariant(
                LinearWithAdaLoRA, 
                lambda a: {"init_r": a.init_r}, 
                ""),
    "use_delta_lora": LoRAVariant(
                LinearWithDeltaLoRA, 
                lambda a: {"update_ratio": a.delta_lora_update_ratio}, 
                ""),
    "use_lora_moe": LoRAVariant(
                LinearWithLoRAMoE, 
                lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, "lora_moe_top_k": a.lora_moe_top_k}, 
                ""),
    "use_milora": LoRAVariant(
                LinearWithMILoRA, 
                lambda a: {"fast_svd_n_iters": a.milora_n_iters}, 
                ". The initialization of milora requires some time especially for full svd decomposition, waiting..."),
    "use_plora": LoRAVariant(
                LinearWithPLoRA, 
                lambda a: {"plora_momentum": a.plora_momentum}, 
                lambda a: f". PLoRA will reset lora weights with momentum: {a.plora_momentum} at every step."),
    "use_mora": LoRAVariant(
                LinearWithMoRA, 
                lambda a: {"mora_type": a.mora_type}, 
                ""),
    "use_gora": LoRAVariant(
                LinearWithGoRA, 
                lambda a: {"gora_init_method": a.gora_init_method,
                            "gora_rank_stablize": a.gora_rank_stablize,
                            "gora_dynamic_scaling": a.gora_dynamic_scaling}, 
                ""),
    "use_increlora": LoRAVariant(
                LinearWithIncreLoRA, 
                lambda a: {"init_r": a.init_r}, 
                ""),
    "use_salora": LoRAVariant(
                LinearWithSALoRA, 
                lambda a: {"init_r": a.init_r, "target_r": a.target_r}, 
                ""),
    "use_mola": LoRAVariant(
                LinearWithMoLA,
                lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, "lora_moe_top_k": a.lora_moe_top_k}, ""),
    "use_nlora": LoRAVariant(
                LinearWithNLoRA,
                lambda a: {"weight_ab_mixer_init_method": None}, 
                ""),
    "use_nora":  LoRAVariant(
                LinearWithNoRA,
                lambda a: {"fast_svd_n_iters": a.nora_n_iters}, 
                ""),
    "use_randlora": LoRAVariant(
                LinearWithRandLoRA, 
                lambda a: {}, 
                ""),
    "use_dude": LoRAVariant(
                LinearWithDude,
                lambda a: {"fast_svd_n_iters":a.pissa_n_iters}, 
                ""),
    "use_loraga_pro": LoRAVariant(
                LinearWithLoRAGAPro,
                lambda a: {"rank_stablize":a.lora_ga_pro_rank_stablize, "dynamic_scaling":a.lora_ga_pro_dynamic_scaling}, 
                "")
}

class LoRAManager:
    """
    Manager class for LoRA operations including layer creation, configuration, and module replacement.
    Provides centralized control over LoRA-related functionalities.
    """
    
    @staticmethod
    def get_lora_layer_class(args: Namespace) -> Tuple[Type, Dict[str, Any], str]:
        """
        Get the appropriate LoRA layer class and its configuration based on input arguments.

        Args:
            args: Namespace containing configuration parameters

        Returns:
            Tuple containing:
                - The LoRA layer class
                - Configuration dictionary for the layer
                - Initialization message
        """
        lora_layer_class = LinearWithLoRA
        variant_config = {}
        variant_message = ""
        
        if getattr(args, "relora_steps", False) or getattr(args, "relora_counts", False):
            variant_message = f". Will reset lora weights every {args.relora_steps} global update steps."
        else:
            for attr_name, variant in LORA_VARIANTS.items():
                if getattr(args, attr_name, False):
                    lora_layer_class = variant.class_type
                    variant_config = variant.config_generator(args)
                    variant_message = variant.init_message(args) if callable(variant.init_message) else variant.init_message
                    break
        
        print_rank_0(f'--->Using lora variant: {lora_layer_class.__name__}{variant_message}', 
                    rank=args.global_rank)
        return lora_layer_class, variant_config

    @staticmethod
    def create_lora_config(module: nn.Module, args: Namespace) -> LoRAConfig:
        """
        Create LoRA configuration for a given module.

        Args:
            module: The neural network module to create configuration for
            args: Namespace containing LoRA parameters

        Returns:
            LoRAConfig object with the specified parameters
        """
        return LoRAConfig(
            lora_rank=args.lora_rank,
            lora_scaler=args.lora_scaler,
            lora_dropout=args.lora_dropout,
            run_lora_in_fp32=args.run_lora_in_fp32,
            weight_a_init_method=args.weight_a_init_method,
            weight_b_init_method=args.weight_b_init_method,
            in_features=module.in_features,
            out_features=module.out_features,
            bias=(getattr(module, "bias", None) is not None),
            quant=getattr(module, "quant", False)
        )

    @staticmethod
    def create_lora_layer(module: nn.Module, 
                         lora_layer_class: Type,
                         variant_config: Dict[str, Any],
                         args: Namespace,
                         transposition: bool = False) -> LinearWithLoRA:
        """
        Create a new LoRA layer instance based on an existing module.

        Args:
            module: Source module to create LoRA layer from
            lora_layer_class: Class type for the LoRA layer
            variant_config: Configuration dictionary for the specific LoRA variant
            args: General configuration arguments
            transposition: Whether to transpose the weight matrix

        Returns:
            Initialized LoRA layer instance
        """
        lora_config = LoRAManager.create_lora_config(module, args)
        lora_layer = lora_layer_class(lora_config, **variant_config)

        # Copy weights
        if transposition:
            lora_layer.weight = nn.Parameter(module.weight.data.T)
        else:
            lora_layer.weight.data = module.weight.data

        # Copy additional attributes
        if getattr(module, "quant", False):
            lora_layer.weight_scaler = module.weight_scaler
        lora_layer.bias = getattr(module, "bias", None)
        
        # Initialize LoRA weights
        lora_layer.init_lora_weights()
        return lora_layer

    @staticmethod
    def should_replace_module(name: str, replace_modules: List[str]) -> bool:
        """
        Determine if a module should be replaced with a LoRA layer.

        Args:
            name: Name of the module
            replace_modules: List of module names to be replaced

        Returns:
            Boolean indicating whether the module should be replaced
        """
        if 'all-linear' in replace_modules and 'lm_head' not in name:
            return True
        return any(module_name in name for module_name in replace_modules)

def switch_to_lora(model: nn.Module, args: Namespace, transposition: bool = False) -> None:
    """
    Replace specified linear layers in the model with LoRA layers.

    Args:
        model: The neural network model to modify
        args: Configuration arguments including LoRA parameters
        transposition: Whether to transpose weight matrices during replacement

    Raises:
        AssertionError: If replace_modules is None
    """
    assert args.replace_modules is not None, 'Replace modules cannot be None'
    
    lora_layer_class, variant_config = LoRAManager.get_lora_layer_class(args)
    if args.run_lora_in_fp32:
        print_rank_0('--->Will keep lora weights in float32', args.global_rank)

    for name, module in model.named_modules():
        try:
            if LoRAManager.should_replace_module(name, args.replace_modules):
                if isinstance(module, LinearWithLoRA):
                    module.merge_and_reset(new_rank=args.rank)
                elif isinstance(module, nn.Module) and all(hasattr(module, attr) 
                    for attr in ["in_features", "out_features", "weight"]):
                    lora_layer = LoRAManager.create_lora_layer(
                        module, lora_layer_class, variant_config, args, transposition
                    )
                    parent = get_parent_model(model, module)
                    if parent:
                        module_name = [k for k, v in parent._modules.items() if v is module][0]
                        setattr(parent, module_name, lora_layer)
            elif isinstance(module, LinearWithLoRA):
                module.merge_and_del()
        except Exception as e:
            print_rank_0(f"Error processing module {name}: {str(e)}", args.global_rank)

def setup_lora(model: nn.Module, args: Namespace, model_config: Optional[Any] = None) -> None:
    """
    Set up LoRA for the model by configuring and applying LoRA layers.

    Args:
        model: The neural network model to apply LoRA to
        args: Configuration arguments including LoRA parameters
        model_config: Optional model configuration containing LoRA settings
    """
    if not args.use_lora:
        return

    # Handle replace_modules parameter
    if args.replace_modules is None:
        args.replace_modules = getattr(model_config, "lora_layers", None)
    elif isinstance(args.replace_modules, str):
        args.replace_modules = args.replace_modules.split('_')

    if args.replace_modules:
        print_rank_0(f'--->LoRA targeting modules: {args.replace_modules}', args.global_rank)
    else:
        print_rank_0('--->The replace modules is not provided, LoRA is targeting all linear modules.', 
                    args.global_rank)
        args.replace_modules = ['all-linear']

    # Apply LoRA
    switch_to_lora(model, args)
    if not check_applied_lora(model):
        print_rank_0(f'--->Cannot find replace modules: {args.replace_modules} in model, '
                    'LoRA is targeting all-linear now.')
        args.replace_modules = ['all-linear']
        switch_to_lora(model, args)

    # Set enable_list
    lora_weight = ['lambda', 'gemma'] if args.use_randlora else (
        ['lambda'] if args.use_vera else (
        ['weight_b'] if args.lora_fa else ['weight_a', 'weight_b']
    ))
    args.enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
    
    model.to(args.device)

def check_applied_lora(model: nn.Module) -> bool:
    """
    Check if LoRA has been applied to any layer in the model.

    Args:
        model: The neural network model to check

    Returns:
        Boolean indicating whether any LoRA layers are present
    """
    return any(isinstance(module, LinearWithLoRA) for module in model.modules())

def recover_linear(model: nn.Module) -> None:
    """
    Recover LoRA layers back to standard linear layers.
    This involves merging LoRA weights and replacing the layer instances.

    Args:
        model: The neural network model containing LoRA layers
    """
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            try:
                module.merge_and_del()
                linear_layer = nn.Linear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=False,
                    dtype=module.weight.dtype,
                    device=module.weight.device
                )
                linear_layer.weight.data = module.weight.data
                
                parent = get_parent_model(model, module)
                if parent:
                    module_name = [k for k, v in parent._modules.items() if v is module][0]
                    setattr(parent, module_name, linear_layer)
            except Exception as e:
                print(f"Error recovering linear layer: {str(e)}")

def get_parent_model(parent_model: nn.Module, module: nn.Module) -> Optional[nn.Module]:
    """
    Recursively find the parent module of a given module in the model.

    Args:
        parent_model: The model to search in
        module: The module to find the parent for

    Returns:
        The parent module if found, None otherwise
    """
    for sub_module in parent_model._modules.values():
        if sub_module is module:
            return parent_model
        if parent := get_parent_model(sub_module, module):
            return parent
    return None