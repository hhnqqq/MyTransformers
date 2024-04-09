def format_param_count(num_params):
    if num_params >= 1e9:
        return f'{num_params / 1e9:.2f}B'
    elif num_params >= 1e6:
        return f'{num_params / 1e6:.2f}M'
    else:
        return str(num_params)

def print_trainable_module_names(model):
    print('--->trainable modules are listed below:')
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_trainable_params = p.numel()
            formatted_params = format_param_count(num_trainable_params)
            print(f'--->module: {name}, trainable parameters: {formatted_params}')

def disable_untrainable_params(model,unable_list):
    for n, p in model.named_parameters():
        flag = False
        for e in unable_list:
            if e.lower() in n.lower():
                flag = True
                break
        if not flag:
            p.requires_grad_(True)

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

def refresh_config(ds_config, args):
    ds_config['gradient_accumulation_steps'] = args.gradient_accumulation_steps
    ds_config['train_micro_batch_size_per_gpu'] = args.batch_size_per_gpu
    ds_config['optimizer']['params']['lr'] = args.lr
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