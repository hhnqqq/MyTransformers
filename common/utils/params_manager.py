from common.utils import print_rank_0
from common.lora_modules import LinearWithLoRA

def format_param_count(num_params):
    if num_params >= 1e9:
        return f'{num_params / 1e9:.2f}B'
    elif num_params >= 1e6:
        return f'{num_params / 1e6:.2f}M'
    else:
        return str(num_params)

def print_trainable_module_names(model, global_rank=0):
    print_rank_0('--->trainable modules are listed below:', global_rank)
    total_params = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_trainable_params = p.numel()
            total_params += num_trainable_params
            formatted_params = format_param_count(num_trainable_params)
            print_rank_0(f'--->Module: {name}, trainable parameters: {formatted_params}', global_rank)
    formatted_total_params = format_param_count(total_params)
    print_rank_0(f'--->Model trainable parameters: {formatted_total_params}', global_rank)

def disable_untrainable_params(model,unable_list):
    for n, p in model.named_parameters():
        flag = False
        for e in unable_list:
            if e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)

def enable_trainable_params(model,enable_list):
    for n, p in model.named_parameters():
        flag = False
        for e in enable_list:
            if e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(False)
        else:
            p.requires_grad_(True)

def set_up_trainable_param(model, args):
    """
    Set up trainable parameters of the model according to `args.enable_list` and `args_diable_list`
    and print trainable paramters.

    `args.enable_list` will be considered at first.
    if `args.enable_list` is None, then `args.diable_list` will be considered.
    if both enable_list and disable_list are None, then all parameters will be set to be trainable.

    For example:
        if `args.enable_list == ['wq']` then wq will be trainable and other weights are not.

        if `args.enable_list is None` and `args.diable_list == ['tok_embeddings']` then tok_embeddings
        will be disabled and other weights are trainable
    """
    if args.enable_list is not None:
        if args.disable_list is not None:
            print_rank_0('--->`args.disable_list` will be ignored as `args.enable_list` is not None.',
                         args.global_rank)
        enable_trainable_params(model, args.enable_list)
    elif args.disable_list is not None:
        disable_untrainable_params(model, args.disable_list)
    else:
        print_rank_0('--->All parameters will be set to trainable as both `args.enable_list` and `args.diable_list` are None',
                     args.global_rank)
        disable_untrainable_params(model, [])
    for module in model.modules():
        if isinstance(module, LinearWithLoRA) and module.weight.requires_grad:
            # If the lora layer's pretrained weight is trainable, disable the lora weight....
            module.weight_a = None
            module.weight_b = None
            del module._parameters['weight_a']
            del module._parameters['weight_b']
    
    if args.num_pp_stages:
        print_trainable_module_names(model)
    else:
        print_trainable_module_names(model, args.global_rank)
        
def refresh_config(ds_config, args):
    """
    Refresh the deepspeed config from args.
    The deepspeed config originally read from static config file.
    """
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size_per_gpu
    ds_config['optimizer']['params']['lr'] = args.lr
    ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
    ds_config["gradient_clipping"] = args.clip_grad_max_norm
    if 'train_batch_size' in ds_config:
        ds_config['train_batch_size'] = args.batch_size_per_gpu * args.gpu_count
    if args.csv_monitor:
        ds_config["csv_monitor"]["enabled"] = True
        ds_config["csv_monitor"]["output_path"] = args.monitor_file_path
        ds_config["csv_monitor"]["job_name"] = args.experiment_name
        ds_config["tensorboard"]["enabled"] = True
        ds_config["tensorboard"]["output_path"] = args.monitor_file_path
        ds_config["tensorboard"]["job_name"] = args.experiment_name
    if args.fp16:
        ds_config["fp16"]["enabled"] = True
        ds_config["bf16"]["enabled"] = False
    elif args.bf16:
        ds_config["fp16"]["enabled"] = False
        ds_config["bf16"]["enabled"] = True
    return ds_config

def set_up_multi_nodes_traning():
    pass

def set_up_model_config_from_args(model_config, args, arg_names):
    arg_names = [arg_name.replace('-',"_") for arg_name in arg_names]
    for arg_name in arg_names:
        arg = getattr(args, arg_name, None)
        setattr(model_config, arg_name, arg)

def set_up_multimodal_config(model_config, args):
    multimodal_arg_names = ["multimodal-projector-type",
                            "multimodal-k-tokens",
                            "multimodal-sample_mode",
                            "multimodal-encode-fp32",
                            "multimodal-projector-layers",
                            "multimodal-sample-mode"]
    set_up_model_config_from_args(model_config, args, multimodal_arg_names)