"""
LoRA and LoRA variants implementation.

Currently suppored LoRA variants are listed below:
1. vanilla LoRA
2. MELoRA [MELoRA: Mini-Ensemble Low-Rank Adapters for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2402.17263)
3. LoRA-GA [LoRA-GA: Low-Rank Adaptation with Gradient Approximation](https://arxiv.org/abs/2407.05000)
4. MosLoRA [Mixture-of-Subspaces in Low-Rank Adaptation](https://arxiv.org/abs/2406.11909)
5. ReLoRA [ReLoRA: High-Rank Training Through Low-Rank Updates](https://arxiv.org/abs/2307.05695)
6. DoRA [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)
7. AdaLoRA [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
8. LoRA-pro [LoRA-Pro: Are Low-Rank Adapters Properly Optimized?](https://arxiv.org/abs/2407.18242)
9. MILoRA 
10. LoRA+
11. PISSA
12. OLoRA
13. EVA
14. LoRA-ga
15. LoRAMoE
16. ReLoRA
17. PLoRA
18. MoRA
19. DeltaLoRA
20. LoRA-FA
21. IncreLoRA [IncreLoRA: Incremental Parameter Allocation Method for Parameter-Efficient Fine-tuning](https://arxiv.org/abs/2308.12043)
22. VeRA [VeRA: Vector-based Random Matrix Adaptation](https://arxiv.org/abs/2310.11454)
23. Shared LoRA - A new variant where A and B matrices are shared across all layers
"""
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
from common.lora_modules.lora_share import *
from common.lora_modules.eva import *

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

def prepare_lora(model, train_dataloader, tokenizer, args):
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
        pad_id = getattr(tokenizer, 'label_pad_id', tokenizer.pad_id)
        EVA_reinit(
            model=model,
            dataloader=train_dataloader,
            args=args,
            pad_id = pad_id
        )
    
    # Prepare shared weights for VeRA
    if (args.use_vera and not args.vera_init_unique_weights) or args.use_lora_share or args.use_randlora or args.use_rasa or args.use_dense_lora:
        
        if args.use_randlora:
            from common.lora_modules.randlora import prepare_shared_lora_weights_randlora as prepare_shared_lora_weights
        if args.use_rasa:
            from common.lora_modules.rasa import prepare_shared_lora_weights_rasa as prepare_shared_lora_weights
            from common.lora_modules.lora_share import update_grouped_shared_weights_to_layer as update_shared_weights_to_layer
        if args.use_dense_lora:
            from common.lora_modules.dense_lora import prepare_shared_lora_weights_denselora as prepare_shared_lora_weights
            from common.lora_modules.lora_share import update_grouped_shared_weights_to_layer as update_shared_weights_to_layer

            
        print_rank_0("Preparing shared LoRA weights...", args.global_rank)
        prepare_shared_lora_weights(model, args)
        update_shared_weights_to_layer(model)
        print_rank_0("Shared LoRA weights prepared and applied.", args.global_rank)