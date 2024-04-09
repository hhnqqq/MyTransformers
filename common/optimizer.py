import warnings
import torch.optim as optim
from common.utils import print_rank_0
from common.scheduler import AnnealingLR
from transformers.utils.versions import require_version
from deepspeed.ops.adam.fused_adam import FusedAdam

def get_optimizer(ds_config, args, model):
    # TODO: Add support to PLoRA and so no ...
    if args.diy_optimizer:

        # Acquire optimizer type.
        if args.optim_type is not None:
            optim_type = args.optim_type.lower()
        elif 'optimizer' in ds_config:
            optim_type = ds_config['optimizer'].get('type', 'adamw').lower()

        # Acquire optimizer.
        if args.use_galore:
            message = 'galore cannot be used with the current DeepSpeed version, and running it will result in an error.'
            warnings.warn(message, UserWarning)
            isSuccess, optimizer =  get_galore_optimizer(optim_type, args, model)
        else:    
            isSuccess, optimizer =  get_regular_optimizer(optim_type, args, model)

        # If successful acquire optimizer, overwrite the default setting in the ds config.
        if isSuccess:
            del ds_config['optimizer']
            print_rank_0(F'--->deepspeed optimizer setting have been overwritten', args.global_rank)
        else:
            print_rank_0(f'--->try to use diy optimizer failed, use the ds setting', args.global_rank)

        lr_scheduler = get_learning_rate_scheduler(optimizer, 0, args)
        return optimizer, lr_scheduler
    
    return None, None

def get_galore_optimizer(optim_type, args, model):
    try:
        assert 'galore' in optim_type, 'when use galore, galore optimizer must be chosen'
        from galore_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor
        optimizer_class = {
            'galore_adamw': GaLoreAdamW,
            'galore_adamw8bit': GaLoreAdamW8bit,
            'galore_adafactor': GaLoreAdafactor
        }.get(optim_type)
        if args.galore_per_layer:
            require_version(">=2.1.0")
            optimizer = register_per_layer_optim(optimizer_class,args,model)
        else:
            param_groups = [{'params': [p for p in model.parameters() if p.requires_grad], 
                            'rank': args.galore_rank, 'update_proj_gap': 200, 'scale': args.galore_scaler, 'proj_type': 'left'}]
            optimizer = optimizer_class(param_groups, lr=args.lr)
        isSuccess = True
    except Exception as e:
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer

def register_per_layer_optim(optimizer_class,args,model):
    optimizer_dict = {}
    def optimizer_hook(p):
        if p.grad is None: 
            return
        optimizer_dict[p].step()
        optimizer_dict[p].zero_grad()
    for n, p in model.named_parameters():
        if p.requires_grad:
            print_rank_0(f'--->set parameter:{n}s optimizer to galore optimizer', args.global_rank)
            optimizer_dict[p] = optimizer_class([{'params': [p], 'rank': args.galore_rank, 
            'update_proj_gap': 200, 'scale': args.galore_scaler, 'proj_type': 'left'}], 
            lr=args.lr, weight_decay=args.weight_decay)
            p.register_post_accumulate_grad_hook(optimizer_hook)
    return None

def get_regular_optimizer(optim_type, args, model):
    try:
        if args.use_lora_plus:
            weight_b_group = [p for n, p in model.named_parameters() if p.requires_grad and 'weight_b' in n]
            base_group = [p for n, p in model.named_parameters() if p.requires_grad and 'weight_b' not in n]
            params = [{'params': weight_b_group, 'lr': args.lora_plus_scaler},
                        {'params': base_group, 'lr': 1}]
            print_rank_0(F'--->lora+ is enabled and the lr of weight b is set to {args.lr * args.lora_plus_scaler}', args.global_rank)
        # elif args.use_dora:
        #     pass
        # elif args.use_plora:
        #     pass
        else:
            params = [p for p in model.parameters() if p.requires_grad]

        optimizer_class = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'adamax': optim.Adamax,
            'sparseadam': optim.SparseAdam,
            'fusedadamw':FusedAdam
        }.get(optim_type)
        
        if optimizer_class is None:
            raise NotImplementedError('only support adam and its variants for now')
        
        optimizer = optimizer_class(params=params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps,
                                    betas=tuple(args.betas))
        isSuccess = True
    except:
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer

def get_learning_rate_scheduler(optimizer, iteration, args):
    init_step = max(iteration - args.auto_warmup_steps, 0)
    if optimizer is not None:
        lr_scheduler = AnnealingLR(optimizer,
                                start_lr=args.lr,
                                warmup_iter=args.num_warmup_steps,
                                num_iters=args.num_update_steps,
                                decay_style=args.lr_decay_style,
                                last_iter=init_step,
                                decay_ratio=args.lr_decay_ratio,
                                auto_warmup_steps=args.auto_warmup_steps,
                                auto_warmup_rate=args.auto_warmup_rate
                                )
    else:
        lr_scheduler = None
    return lr_scheduler