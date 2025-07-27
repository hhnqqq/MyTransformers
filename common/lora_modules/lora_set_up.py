import traceback
import torch.nn as nn
from typing import Type, Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass
from argparse import Namespace

from common.utils import print_rank_0
from common.lora_modules.lora import *
from common.lora_modules.dora import LinearWithDoRA
from common.lora_modules.hira import LinearWithHiRA
from common.lora_modules.melora import LinearWithMELoRA
from common.lora_modules.lora_ga import LinearWithLoRAGA
from common.lora_modules.mos_lora import LinearWithMosLoRA
from common.lora_modules.rslora import LinearWithRSLoRA
from common.lora_modules.pissa import LinearWithPiSSA
from common.lora_modules.olora import LinearWithOLoRA
from common.lora_modules.vera import LinearWithVeRA
from common.lora_modules.share_lora import LinearWithShareLoRA
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
from common.lora_modules.lora_one import LinearWithLoRAOne
from common.lora_modules.goat import LinearWithGOAT
from common.lora_modules.rasa import LinearWithRASA
from common.lora_modules.dense_lora import LinearWithDenseLoRA
from common.lora_modules.eva import LinearWithEVA
from common.lora_modules.delora import LinearWithDeLoRA
from common.lora_modules.nzlora import LinearWithNZLoRA
from common.lora_modules.rasa_moe import LinearWithRASAMOE
from common.lora_modules.lora_sb import LinearWithLoRASB
from common.lora_modules.lora_da import LinearWithLoRADA

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
                ""
    ),
    "use_hira": LoRAVariant(
                LinearWithHiRA, 
                lambda a: {}, 
                "HiRA utilizes hadamard production of pre-trained weight and lora weight to increase the overall rank of update."
                "A large learning rate for HiRA is recommended."
    ),
    "use_mos_lora": LoRAVariant(
                LinearWithMosLoRA, 
                lambda a: {"weight_ab_mixer_init_method": a.weight_ab_mixer_init_method}, 
                "MosLoRA introduces a mixer matrix to mix features of A and B leading to a stronger expressive ability."
    ),
    "use_me_lora": LoRAVariant(
                LinearWithMELoRA, 
                lambda a: {"me_lora_n_split": a.me_lora_n_split, "forward_method": a.me_lora_forward_method}, 
                "MeLoRA introduces a block diagonal structure to increase the overall rank."
    ),
    "use_lora_ga": LoRAVariant(
                LinearWithLoRAGA, 
                lambda a: {}, 
                lambda a: "LoRA-GA utilizes SVD to extract singular features of gradient of pre-trained weight "
                "to initialize low-rank weights, accelerate the convergence. The initialization of LoRA-GA requires some time, "
                f"which depends on the number of gradient computing steps: {a.gradient_est_n_steps}"
    ),
    "use_lora_one": LoRAVariant(
                LinearWithLoRAOne, 
                lambda a: {}, 
                lambda a: "LoRA-One utilizes SVD to extract singular features of gradient of pre-trained weight "
                "to initialize low-rank weights, accelerate the convergence. The initialization of LoRA-One requires some time, "
                f"which depends on the number of gradient computing steps: {a.gradient_est_n_steps}"
    ),
    "use_rslora": LoRAVariant(
                LinearWithRSLoRA, 
                lambda a: {}, 
                "RSLoRA introduces a root square scaling for LoRA, stablizing the training process."
    ),
    "use_pissa": LoRAVariant(
                LinearWithPiSSA, 
                lambda a: {"fast_svd_n_iters": a.pissa_n_iters, "keep_init_weights": a.pissa_keep_init_weights},
                "PiSSA utilizes SVD to extract singular features of pre-trained weight "
                "to initialize low-rank weights, accelerate the convergence. "
                "The initialization of PiSSA requires some time especially for full svd decomposition, waiting..."
    ),
    "use_olora": LoRAVariant(
                LinearWithOLoRA, 
                lambda a: {}, 
                "OLoRA utilizes QR decomposition to extract singular features of pre-trained weight "
                "to initialize low-rank weights, accelerate the convergence. "
                "The initialization of OLoRA requires some time, waiting..."
    ),
    "use_vera": LoRAVariant(
                LinearWithVeRA, 
                lambda a: {"lambda_b_init_method":a.lambda_b_init_method, "lambda_d_init_method":a.lambda_d_init_method, "init_unque_lora_weights":a.vera_init_unique_weights}, 
                "VeRA shares A and B across layers if not args.vera_init_unique_weights, and keeps A and B frozen during training process. "
                "Only vector weights are tuned during training. Enabling a larger rank compared to LoRA under same resource constraints."
    ),
    "use_sharelora": LoRAVariant(
                LinearWithShareLoRA,
                lambda a: {"share_part": a.sharelora_share_part},
                "Shared LoRA shares A and B matrices across all layers. Both A and B matrices are trainable. "
                "This significantly reduces parameters while maintaining expressiveness."
    ),
    "use_tied_lora": LoRAVariant(
                LinearWithVeRA, 
                lambda a: {"lambda_b_init_method":a.lambda_b_init_method, "lambda_d_init_method":a.lambda_d_init_method,}, 
                "Tied-LoRA shares A and B across layers. "
                "A, B and vector weights are tuned during training. Enabling a larger rank compared to LoRA under same resource constraints."
    ),
    "use_adalora": LoRAVariant(
                LinearWithAdaLoRA, 
                lambda a: {"init_r": a.init_r}, 
                "AdaLoRA enables adaptive rank allocation by masking relatively un-important ranks during training."
    ),
    "use_delta_lora": LoRAVariant(
                LinearWithDeltaLoRA, 
                lambda a: {"update_ratio": a.delta_lora_update_ratio}, 
                ""
    ),
    "use_lora_moe": LoRAVariant(
                LinearWithLoRAMoE, 
                lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, "lora_moe_top_k": a.lora_moe_top_k}, 
                ""
    ),
    "use_milora": LoRAVariant(
                LinearWithMILoRA, 
                lambda a: {"fast_svd_n_iters": a.milora_n_iters}, 
                "MILoRA utilizes SVD to extract the least singular features of pre-trained weight "
                "to initialize low-rank weights, accelerate the convergence. "
                "The initialization of milora requires some time, waiting..."
    ),
    "use_plora": LoRAVariant(
                LinearWithPLoRA, 
                lambda a: {"plora_momentum": a.plora_momentum}, 
                lambda a: f"PLoRA will reset lora weights with momentum: {a.plora_momentum} at every step."
    ),
    "use_mora": LoRAVariant(
                LinearWithMoRA, 
                lambda a: {"mora_type": a.mora_type}, 
                ""
    ),
    "use_gora": LoRAVariant(
                LinearWithGoRA, 
                lambda a: {"gora_init_method": a.gora_init_method,
                "gora_rank_stablize": a.gora_rank_stablize,
                "gora_dynamic_scaling": a.gora_dynamic_scaling}, 
                lambda a: "GoRA utilize gradient of pre-trained weight to allocate rank and intialize weights for low-rank adapters. "
                "accelerate the convergence. The initialization of GoRA requires some time, "
                f"which depends on the number of gradient computing steps: {a.gradient_est_n_steps}"
    ),
    "use_increlora": LoRAVariant(
                LinearWithIncreLoRA, 
                lambda a: {"init_r": a.init_r}, 
                "IncreLoRA adaptively increse rank of LoRA during training."
    ),
    "use_salora": LoRAVariant(
                LinearWithSALoRA, 
                lambda a: {"init_r": a.init_r, "target_r": a.target_r}, 
                ""
    ),
    "use_mola": LoRAVariant(
                LinearWithMoLA,
                lambda a: {"lora_moe_n_experts": a.lora_moe_n_experts, "lora_moe_top_k": a.lora_moe_top_k}, ""
    ),
    "use_nlora": LoRAVariant(
                LinearWithNLoRA,
                lambda a: {"weight_ab_mixer_init_method": None}, 
                ""
    ),
    "use_nora":  LoRAVariant(
                LinearWithNoRA,
                lambda a: {"fast_svd_n_iters": a.nora_n_iters}, 
                ""
    ),
    "use_randlora": LoRAVariant(
                LinearWithRandLoRA, 
                lambda a: {}, 
                "RandLoRA increases the overall rank by introducing multiple AB pairs to a layer, and only tune vector weights."
    ),
    "use_dude": LoRAVariant(
                LinearWithDude,
                lambda a: {"fast_svd_n_iters":a.pissa_n_iters}, 
                "Dude combine DoRA and PiSSA together "
                "The initialization of PiSSA requires some time especially for full svd decomposition, waiting..."
    ),
    "use_loraga_pro": LoRAVariant(
                LinearWithLoRAGAPro,
                lambda a: {"rank_stablize":a.lora_ga_pro_rank_stablize, "dynamic_scaling":a.lora_ga_pro_dynamic_scaling}, 
                ""
    ),
    "use_goat": LoRAVariant(
                LinearWithGOAT,
                lambda a: {"scaling_type":a.goat_scaling_type,
                "init_type":a.goat_init_type,
                "n_experts":a.lora_moe_n_experts,
                "top_k":a.lora_moe_top_k,
                "rho":a.goat_rho,
                "eta":a.goat_eta,
                "init_cof":a.goat_init_cof},
                "The initialization of GOAT requires some time, waiting..."
    ),
    "use_rasa": LoRAVariant(
                LinearWithRASA,
                lambda a: {"shared_lora_rank":a.rasa_shared_lora_rank},
                "RASA shares some lora ranks across layers to increase the overall rank."
    ),
    "use_dense_lora": LoRAVariant(
                LinearWithDenseLoRA,
                lambda a: {},
                "DenseLoRA is similar to MosLoRA while weight_a and weight_b are shared across layers."
                "DenseLoRA also introduce non-linear function for LoRA computation."
    ),
    "use_eva": LoRAVariant(
                LinearWithEVA,
                lambda a: {},
                "EVA is a variant of LoRA, which uses the SVD decomposition result of the activation values to initialize the A matrix weights of LoRA."
    ),
    "use_delora": LoRAVariant(
                LinearWithDeLoRA,
                lambda a: {"delora_lambda": a.delora_lambda},
                "DeLoRA bounding the distance of the transformation, effectively decouples the angular learning from the adaptation strength,"
                "enhancing robustness without compromising performance."
    ),
    "use_nzlora": LoRAVariant(
                LinearWithNZLoRA,
                lambda a: {"reset_weight": a.lora_reset_weight, "init_scale_a": a.nzlora_init_scale_a, "init_scale_b": a.nzlora_init_scale_b},
                "NZLoRA use kaiming uniform to initialize weight_a and weight_b."
    ),
    "use_rasamoe": LoRAVariant(
                LinearWithRASAMOE,
                lambda a: {"num_experts": a.lora_moe_n_experts, 
                "top_k":a.lora_moe_top_k, 
                "shared_lora_rank":a.rasa_shared_lora_rank},
                "Rasamoe is divided into fixed experts and shared experts."
    ),
    "use_lora_sb": LoRAVariant(
                LinearWithLoRASB,
                lambda a: {},
                "LoRA-SB is similar to LoRA-GA and MoSlLoRA, during training only weight_ab_mixer reamins trainable."
    ),
    "use_lora_da":LoRAVariant(
                LinearWithLoRADA,
                lambda a: {},
                ""
    ),

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
        
        print_rank_0(f'--->Using lora variant: {lora_layer_class.__name__}. {variant_message}', 
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
            bias=(getattr(module, "bias", None) is not None)
        )
    
    @staticmethod
    def check_lora_settings(args, lora_layer_class):
        # Check incompatible settings
        if any(getattr(args, attr, False) for attr in ['use_dora', 'use_dude', 'use_hira', 'use_delta_lora']) and args.lora_dropout:
            print_rank_0(
                f'LoRA dropout is not compatible with class: {lora_layer_class.__name__}, skip',
                args.global_rank
            )
        
        # Validate GOAT settings
        if getattr(args, "use_goat", False):
            valid_scaling_types = {'lora', 'rslora', 'goat'}
            if args.goat_scaling_type not in valid_scaling_types:
                raise ValueError(
                    f"Invalid scaling type for goat: {args.goat_scaling_type}. "
                    f"Choose from {sorted(valid_scaling_types)}."
                )
            
            valid_init_types = {'goat', 'goat-mini', 'vanilla'}
            if args.goat_init_type not in valid_init_types:
                raise ValueError(
                    f"Invalid initialization type for goat: {args.goat_init_type}. "
                    f"Choose from {sorted(valid_init_types)}."
                )
            
        if any(getattr(args, attr, False) for attr in ['use_lora_moe', 'use_goat', 'use_rasamoe']):
            if args.lora_moe_n_experts < args.lora_moe_top_k:
                raise ValueError(
                    f"top_k < n_experts is expected for moe based lora variants, got {args.lora_moe_n_experts} and {args.lora_moe_top_k}."
                )
        
        # Validate GORA settings
        if getattr(args, "use_gora", False):
            valid_gora_methods = {'vanilla', 'weight_svd', 'grad_svd', 'compress'}
            if args.gora_init_method not in valid_gora_methods:
                raise ValueError(
                    f"Invalid initialization type for gora: {args.gora_init_method}. "
                    f"Choose from {sorted(valid_gora_methods)}."
                )
        
        # Validate VERA settings
        if any([args.use_vera, args.use_delora]) and args.weight_b_init_method is None:
            raise ValueError(f'The init method for weight b cannot be None when {lora_layer_class.__name__} is applied.')
        
        if args.use_mora and args.mora_type not in {'rope', 'sharing'}:
            raise ValueError(f'Not supported mora type: {args.mora_type}!')
                
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
        lora_layer.bias = getattr(module, "bias", None)
        
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
    if args.replace_modules is None:
        raise ValueError('Replace modules cannot be None')
    
    lora_layer_class, variant_config = LoRAManager.get_lora_layer_class(args)
    LoRAManager.check_lora_settings(args, lora_layer_class)
    if args.run_lora_in_fp32:
        print_rank_0('--->The low-rank weights will be cast into FP32 during forward computation.', args.global_rank)

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
        except Exception:
            e = traceback.format_exc()
            print_rank_0(f"Error processing module {name}: {e}", args.global_rank)

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
        print_rank_0(f'--->LoRA target modules identified: {args.replace_modules}', 
                    args.global_rank)
    else:
        print_rank_0('--->No specific LoRA target modules provided - defaulting to all linear layers', 
                    args.global_rank)
        args.replace_modules = ['all-linear']

    # Apply LoRA
    print_rank_0('--->Converting target modules to LoRA layers', args.global_rank)
    switch_to_lora(model, args)
    
    if not check_applied_lora(model):
        print_rank_0(f'--->Specified modules {args.replace_modules} not found - '
                    'falling back to all linear layers', args.global_rank)
        args.replace_modules = ['all-linear']
        switch_to_lora(model, args)

    # Configure enabled parameters
    lora_weight_names = get_lora_weight_names(args)
    args.enable_list = lora_weight_names if args.enable_list is None else list(set(args.enable_list + lora_weight_names))
    
    print_rank_0(f'--->Successfully initialized the model with LoRA modules.',
                args.global_rank)
    model.to(args.device)

def get_lora_weight_names(args):
    conditions = [
        (args.use_randlora, ['lambda', 'gemma']),
        (args.use_vera, ['lambda']),
        (args.lora_fa, ['weight_b']),
        (args.use_tied_lora or args.use_delora, ['weight_a', 'weight_b', 'lambda']),
        (args.use_dora or args.use_dude, ['weight_a', 'weight_b', 'origin_magnitude']),
        (args.use_adalora or args.use_rasa, ['weight_a', 'weight_b', 'weight_e']),
        (args.use_mos_lora or args.use_dense_lora or args.use_nlora, ['weight_a', 'weight_b', 'weight_ab_mixer']),
        (args.use_goat or args.use_lora_moe or args.use_rasamoe, ['weight_a', 'weight_b', 'gate']),
        (args.use_lora_sb, ['weight_ab_mixer']),
        (True, ['weight_a', 'weight_b'])
    ]

    return next(value for condition, value in conditions if condition)

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