"""
LoRA and LoRA variants implementation.
"""
import re
import inspect
import contextlib
from common.lora_modules.lora import *
from common.lora_modules.melora import *
from common.lora_modules.lora_ga import *
from common.lora_modules.lora_set_up import *
from common.lora_modules.mos_lora import *
from common.lora_modules.dora import *
from common.lora_modules.lorapro_optim import *
from common.lora_modules.lora_moe import *
from common.lora_modules.plora import * 
from common.lora_modules.adalora import *
from common.lora_modules.gora import *
from common.lora_modules.increlora import *
from common.lora_modules.mola import *
from common.lora_modules.dude import *
from common.lora_modules.lora_ga_pro import *
from common.lora_modules.goat import *
from common.lora_modules.lora_one import *
from common.lora_modules.vera import *
from common.lora_modules.eva import *
from common.lora_modules.rasa_moe import *

@contextlib.contextmanager
def DisableLoRA(model):
    """
    Context manager to disable LoRA functionality for all LinearWithLoRA layers in the model.

    Args:
        model: The PyTorch model containing LinearWithLoRA layers.

    Usage:
        with DisableLoRA(model):
            # LoRA is disabled within this block
            output = model(input)
    """
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module.disable_lora = True

    try:
        yield
    finally:
        for module in model.modules():
            if isinstance(module, LinearWithLoRA):
                module.disable_lora = False

@contextlib.contextmanager
def MergeLoRA(model):
    for module in model.modules():
        if isinstance(module, LinearWithLoRA):
            module._merge_lora()

    try:
        yield
    finally:
        for module in model.modules():
            if isinstance(module, LinearWithLoRA):
                module._unmerge_lora()

def check_shared_lora_weights_required(args):
    shared_weight_conditions = [
        getattr(args, 'use_vera', False) and not getattr(args, 'vera_init_unique_weights', False),
        getattr(args, 'use_sharelora', False),
        getattr(args, 'use_randlora', False),
        getattr(args, 'use_rasa', False),
        getattr(args, 'use_dense_lora', False),
        getattr(args, 'use_rasamoe', False)
    ]
    
    # Return True if any condition is met
    return any(shared_weight_conditions)

def insert_shared_lora_weights(model, args):
    if args.params_to_save:
        args.params_to_save.append('shared')
    else:
        args.params_to_save = ['shared']

    from common.lora_modules.share_lora import prepare_shared_lora_weights, update_shared_weights_to_layer
    
    if getattr(args, 'use_randlora', False):
        from common.lora_modules.randlora import prepare_shared_lora_weights_randlora as prepare_shared_lora_weights
    if getattr(args, 'use_rasa', False):
        from common.lora_modules.rasa import prepare_shared_lora_weights_rasa as prepare_shared_lora_weights
        from common.lora_modules.share_lora import update_grouped_shared_weights_to_layer as update_shared_weights_to_layer
    if getattr(args, 'use_dense_lora', False):
        from common.lora_modules.dense_lora import prepare_shared_lora_weights_denselora as prepare_shared_lora_weights
        from common.lora_modules.share_lora import update_grouped_shared_weights_to_layer as update_shared_weights_to_layer
    if getattr(args, 'use_rasamoe', False):
        from common.lora_modules.rasa_moe import prepare_shared_lora_weights_rasa as prepare_shared_lora_weights
        from common.lora_modules.share_lora import update_grouped_shared_weights_to_layer as update_shared_weights_to_layer

        
    print_rank_0("--->Preparing shared LoRA weights...", args.global_rank)
    shared_weight_a, shared_weight_b = prepare_shared_lora_weights(model, args)
    update_shared_weights_to_layer(model, shared_weight_a, shared_weight_b)
    print_rank_0("--->Shared LoRA weights prepared and applied.", args.global_rank)

def create_shared_weight_references(model):
    ref_a, ref_b = None, None
    for module in model.modules():
        if isinstance(module, LinearWithLoRA) and getattr(module, "share_lora_weights", False) and module.has_lora_weights:
            ref_a, ref_b = module.shared_weight_a, module.shared_weight_b
            break
    
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA) and getattr(module, "share_lora_weights", False):
            pattern = re.compile(r'layers\.(\d+)\.(.+)')
            match = pattern.search(name)
            
            update_method = module.update_shared_weights
            sig = inspect.signature(update_method)
            params = sig.parameters
            
            if match:
                module_name = match.group(2).replace('.', '__')
                if 'module_name' in params:
                    update_method(ref_a, ref_b, module_name)
                else:
                    update_method(ref_a, ref_b)

def prepare_lora(model, train_dataloader, args):
    """
    Prepare lora if needed

    For example, if LoRA-GA is utilized, then we need to pre-compute gradients before training.
    For VeRA and Shared LoRA, we need to prepare shared weights.
    """
    if args.use_lora_ga:
        lora_ga_reinit(model=model,
                    dataloader=train_dataloader,
                    args=args,
                    iters=args.lora_ga_n_steps)
    if args.use_lora_one:
        lora_one_reinit(model=model,
                    dataloader=train_dataloader,
                    args=args,
                    iters=args.lora_one_n_steps)
    if args.use_gora:
        gora_reinit(model=model,
                    dataloader=train_dataloader,
                    args=args,
                    iters=args.gora_n_steps)
    if args.use_adalora:
        rank_allocator = RankAllocator(model, args)
        model.rankallocator = rank_allocator
    if args.use_increlora:
        rank_allocator = IncreRankAllocator(model, args)
        model.rankallocator = rank_allocator
    if args.use_mola:
        init_mola_experts_by_shape(model=model, args=args)
    if args.use_lora_ga_pro:
        lora_ga_pro_reinit(model=model,
                    dataloader=train_dataloader,
                    args=args,
                    iters=args.lora_ga_n_steps)
    if args.use_eva:
        eva_reinit(
            model=model,
            dataloader=train_dataloader,
            args=args
        )
    
    # Prepare shared weights for VeRA
    if check_shared_lora_weights_required(args):
        insert_shared_lora_weights(model, args)

def prepare_lora_for_inference(model, args):
    switch_to_lora(model, args)
    # Common variables for both conditions
    rank_config = None
    if any(getattr(args, attr, False) for attr in ['use_gora', 'use_eva', 'use_loraga_pro']):
        rank_config_file = os.path.join(args.output_path, args.experiment_name, 'rank.json')
        rank_config = json.load(open(rank_config_file, 'r'))
    
    # Process GoRA modules
    if getattr(args, 'use_gora', False):
        for name, module in model.model.named_modules():
            if isinstance(module, LinearWithGoRA):
                module.init_method = 'vanilla'
                module.dynamic_init(args.lora_rank, rank_config[name])
    
    if getattr(args, 'use_eva', False):
        for name, module in model.model.named_modules():
            if isinstance(module, LinearWithEVA):
                module.EVA_init(rank_config[name])

    # Process LoRA GA Pro modules
    if getattr(args, 'use_lora_ga_pro', False):
        for name, module in model.model.named_modules():
            if isinstance(module, LinearWithLoRAGAPro):
                module.prepare_init(allocated_rank=rank_config[name])
                module.init_lora_weights()
    
    # Check for shared weights
    if check_shared_lora_weights_required(args):
        insert_shared_lora_weights(model, args)