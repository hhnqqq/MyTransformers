import logging
import traceback
import torch.optim as optim
import deepspeed.ops as ds_optim

from functools import partial
from traceback import format_exc
from common.utils import print_rank_0
from common.scheduler import AnnealingLR
from common.lora_modules import LoRAProAdamW
from transformers.utils.versions import require_version

def get_optimizer_type(args, ds_config):
    if args.optim_type is not None:
        return args.optim_type.lower()
    elif 'optimizer' in ds_config:
        return ds_config['optimizer'].get('type', 'adamw').lower()
    return None

def get_optimizer_instance(optim_type, args, model):
    if args.use_galore:
        message = 'galore cannot be used with the current DeepSpeed version, and running it will result in an error.'
        print_rank_0(message, level=logging.ERROR, rank=args.global_rank)
        return get_galore_optimizer(optim_type, args, model)
    elif args.use_lora_pro:
        message = '--->You are using lorapro-adamw optmizer'
        print_rank_0(message, args.global_rank)
        return get_lorapro_optimizer(optim_type, args, model)
    elif args.use_increlora:
        message = '--->You are using increlora optmizer'
        print_rank_0(message, args.global_rank)
        return get_increlora_optimizer(optim_type, args, model)
    else:
        return get_regular_optimizer(optim_type, args, model)

def get_optimizer(ds_config, args, model, optimizer_sd = None, lr_scheduler_sd = None):
    """
    Set up optimizer and learning rate scheduler.

    If `args.diy_optimizer == True` then optimzer will be created according to args.
    else deepseed will create optimizer for you according to ds_config.

    This function provide clear optimizer prepare process and can adjust the parameter groups if needed.
    """
    if not args.diy_optimizer:
        return None, None

    optim_type = get_optimizer_type(args, ds_config)
    offload_config = ds_config["zero_optimization"].get("offload_optimizer", {})
    offload_device = offload_config.get("device", None)
    if offload_device == 'cpu' or args.offload_optimizer:
        optim_type = 'cpu' + optim_type
    isSuccess, optimizer = get_optimizer_instance(optim_type, args, model)

    if isSuccess:
        if 'optimizer' in ds_config:
            del ds_config['optimizer']
        print_rank_0(f'--->Deepspeed optimizer setting has been overwritten', args.global_rank)
    else:
        print_rank_0(f'--->Try to use diy optimizer failed, use the ds setting', args.global_rank)
        return None, None

    lr_scheduler = get_learning_rate_scheduler(optimizer, 0, args)

    if all([optimizer, lr_scheduler, optimizer_sd, lr_scheduler_sd]):
        optimizer.load_state_dict(optimizer_sd)
        lr_scheduler.load_state_dict(lr_scheduler_sd)
    elif any([optimizer_sd, lr_scheduler_sd]):
        print_rank_0(f'--->Optimizer state dict and lr scheduler state dict have not been loaded as optimizer or lr scheduler is None', args.global_rank)

    return optimizer, lr_scheduler
    

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
        lr_group_patterns = set(args.lr_group_patterns) if args.lr_group_patterns else set()
        lr_group_scales = set(args.lr_group_scales) if args.lr_group_scales else set()

        if args.use_lora_plus:
            lr_group_patterns.add('weight_b')
            lr_group_scales.add(args.lora_plus_scaler)
            print_rank_0(F'--->lora+ is enabled and the lr of weight b is set to {args.lr * args.lora_plus_scaler}', args.global_rank)

        if lr_group_patterns and lr_group_scales:
            param_groups = {}
            param_groups["default"] = {"params": [], "lr": 1}
            
            for pattern, lr_scale in zip(lr_group_patterns, lr_group_scales):
                param_groups[pattern] = {"params": [], "lr": lr_scale}
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                matched = False
                for pattern in lr_group_patterns:
                    if pattern in name:
                        param_groups[pattern]["params"].append(param)
                        matched = True
                        break
                
                if not matched:
                    param_groups["default"]["params"].append(param)
            
            params = [group for group in param_groups.values() if group["params"]]
        else:
            params = [{'params':[p for p in model.parameters() if p.requires_grad], 'lr': 1}]

        optimizer_class = {
            'adamw': partial(ds_optim.adam.FusedAdam, adam_w_mode=True),
            'adam': partial(ds_optim.adam.FusedAdam, adam_w_mode=False),
            'cpuadamw':partial(ds_optim.adam.DeepSpeedCPUAdam, adamw_mode=True),
            'cpuadam':partial(ds_optim.adam.DeepSpeedCPUAdam, adamw_mode=False),
            'adamax': optim.Adamax,
            'sparseadam': optim.SparseAdam,
            'torchadam': optim.Adam,
            'torchadamw': optim.AdamW
        }.get(optim_type)
        
        if optimizer_class is None:
            raise NotImplementedError('only support adam and its variants for now')
        
        optimizer = optimizer_class(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps,
                                    betas=tuple(args.betas))
        isSuccess = True
    except Exception:
        print_rank_0(f'--->Load local optimizer error as: {traceback.format_exc()}', args.global_rank)
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer

def get_lorapro_optimizer(optim_type, args, model):
    try:
        if args.use_lora_plus:
            lora_plus_scaler = args.lora_plus_scaler
        else:
            lora_plus_scaler = 1
        named_params = {'params' : ((n, p) for n, p in model.named_parameters() if p.requires_grad)}
        
        optimizer = LoRAProAdamW(named_params,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                eps=args.eps,
                                betas=tuple(args.betas),
                                lora_plus_scaler=lora_plus_scaler)
        isSuccess = True
    except Exception:
        e = format_exc()
        print_rank_0(f'--->Load local optimizer error as e: {e}', args.global_rank)
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer

def get_increlora_optimizer(optim_type, args, model):
    try:
        params = [{'params':[p for p in model.parameters() if p.requires_grad], 'lr': 1}]

        optimizer_class = {
            'adamw': partial(ds_optim.adam.FusedAdam, adam_w_mode=True),
            'adam': partial(ds_optim.adam.FusedAdam, adam_w_mode=False),
        }.get(optim_type)

        if optimizer_class is None:
            raise NotImplementedError('`get_increlora_optimizer` only support adam and its variants for now')
        
        optimizer = optimizer_class(params,
                                    lr=args.lr,
                                    weight_decay=args.weight_decay,
                                    eps=args.eps,
                                    betas=tuple(args.betas))
        isSuccess = True
    except Exception:
        print_rank_0(f'--->Load local optimizer error as: {traceback.format_exc()}', args.global_rank)
        isSuccess = False
        optimizer = None
    return isSuccess, optimizer

def get_learning_rate_scheduler(optimizer, iteration, args):
    init_step = max(iteration - args.auto_warmup_steps, 0)
    if optimizer is not None:
        lr_scheduler = AnnealingLR(optimizer,
                                start_lr=args.lr,
                                warmup_iter=args.num_warmup_steps,
                                num_iters=args.num_global_update_steps,
                                decay_style=args.lr_decay_style,
                                last_iter=init_step,
                                decay_ratio=args.lr_decay_ratio,
                                auto_warmup_steps=args.auto_warmup_steps,
                                auto_warmup_rate=args.auto_warmup_rate,
                                restart_every=args.relora_steps,
                                restart_warmup_steps=args.relora_warmup_steps,
                                global_rank=args.global_rank
                                )
    else:
        lr_scheduler = None
    return lr_scheduler