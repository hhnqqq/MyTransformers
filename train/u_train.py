import os
import json
import torch
from datetime import datetime

from model import *
from common.lora_modules import *
from common.parser import get_args
from common.registry import registry
from common.optimizer import get_optimizer
import common.utils.parallel_states as parallel_states
from train.load_data import load_dataloder
from train.load_model import load_model
from dataset_classes import RepeatingLoader
from common.utils.params_manager import refresh_config, set_up_trainable_param
from common.utils import print_rank_0, read_config, set_random_seed, init_distributed_model, GPUMemoryPrinter

args = get_args()
args = registry.get_paths(args)
set_random_seed(args.seed)

print_rank_0(f'--->Data parallel world size: {parallel_states.get_data_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Sequence parallel world size: {parallel_states.get_sequence_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Pipeline parallel world size: {parallel_states.get_pipeline_model_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Registry contains {json.dumps(registry.list_all(), indent=4, ensure_ascii=False)}', args.global_rank)

with GPUMemoryPrinter('Loading Model', args.global_rank):
    model, tokenizer, model_config, return_dataset_kwargs = load_model(args)
print_rank_0(f'--->Using model class: {model.__class__.__name__}', args.global_rank)
print_rank_0(f'--->Model architecture overview: {model}', args.global_rank)

with GPUMemoryPrinter('Setup LoRA', args.global_rank):
    setup_lora(model, args, model_config)

"""
GPUs=8 sp=4 pp=1 tp=1 dp=2
In this case rank group [0,1,2,3] share the same data sample, and split on BaseModel.cut_sequence()

GPUs=8 sp=1 pp=1 tp=1 dp=8
In this case non of the ranks share the same data sample.

dp_rank parameter controls who share same data sample.
"""
dp_rank = parallel_states.get_data_parallel_rank()
num_dp_rank = parallel_states.get_data_parallel_world_size()
train_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_rank, return_dataset_kwargs, True)
eval_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_rank, return_dataset_kwargs, False)

ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

# start tranning
# Run this befor set up trainable parameters.
prepare_lora(model, train_dataloader, args)
# set up trainable before acquiring optimizer.
set_up_trainable_param(model, args)

with GPUMemoryPrinter('Loading optimizer', args.global_rank):
    optimizer_sd, lr_scheduler_sd = getattr(model_config, 'optmizer_sd',None), getattr(model_config, 'lr_scheduler_sd',None)
    optimizer, lr_scheduler = get_optimizer(ds_config=ds_config, 
                                            args=args, 
                                            model=model, 
                                            optimizer_sd=optimizer_sd, 
                                            lr_scheduler_sd=lr_scheduler_sd)

engine, optimizer, lr_scheduler = init_distributed_model(args, 
                                                         model, 
                                                         optimizer, 
                                                         lr_scheduler, 
                                                         ds_config, 
                                                         parallel_states)


if __name__ == '__main__':

    import wandb
    import logging
    import traceback
    import torch.profiler as profiler

    from torch.profiler import ProfilerActivity
    from train.pp_train import *
    from train.dp_train import *
    from train.trainer import Trainer

    def get_writer(args):
        current_time = datetime.now().strftime('%y-%m-%d_%H-%M')
        if not args.test_code and args.global_rank == 0:
            if args.wandb:
                os.environ['WANDB_CACHE_DIR'] = args.wandb_cache_dir
                os.environ['WANDB_DIR'] = args.wandb_dir
                if args.wandb_api_key:
                    os.environ['WANDB_API_KEY'] = args.wandb_api_key
                wandb.init(project=args.wandb_project,
                        entity=args.wandb_team,
                        mode=args.wandb_mode,
                        name=args.experiment_name + current_time,
                        config=args)
            elif args.tensorboard:
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError:
                    from tensorboard import SummaryWriter
                log_dir = os.path.join(args.tb_log_dir, args.experiment_name + current_time)
                return SummaryWriter(log_dir=log_dir)
            return None

    writer = get_writer(args)
    if args.num_pp_stages:
        forward_step = forward_step_pipeline
        eval_step = eval_step_pipeline
        backward_step = None
        task_print = task_print_pipeline
    else:
        forward_step = forward_step_deepspeed
        eval_step = eval_step_deepspeed
        if args.disable_zero_optimizer and not args.use_increlora:
            backward_step = backward_step_deepspeed_stage0
        elif args.relora_steps is not None:
            backward_step = backward_step_deepspeed_relora
        elif args.use_plora:
            backward_step = backward_step_deepspeed_plora
        elif args.use_delta_lora:
            backward_step = backward_step_deepspeed_deltalora
        elif args.use_adalora:
            backward_step = backward_step_deepspeed_adalora
        elif args.use_increlora:
            backward_step = backward_step_deepspeed_increlora_stage0
        elif args.use_goat or args.use_rasamoe:
            print_rank_0(f"Using aux_ loss for moe model, lora_moe_aux_loss_coeff={args.lora_moe_aux_loss_coeff}", args.global_rank)
            backward_step = backward_step_deepspeed_loramoe
        else:
            backward_step = backward_step_deepspeed
        task_print = task_print_ntp
    
    trainer = Trainer(args, writer)
        
    trainer.register_task_print(task_print)

    def train_with_profiler(profiler):
        trainer.train(
            model=engine,
            train_data_loader=RepeatingLoader(train_dataloader),
            eval_data_loader=None if eval_dataloader is None else RepeatingLoader(eval_dataloader),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            forward_step=forward_step,
            backward_step=backward_step,
            eval_step=eval_step,
            profiler=profiler,
            log_loss=True
        )

    try:
        profiler = None
        if args.profile_log_dir:
            log_dir = os.path.join(args.profile_log_dir, f"{args.experiment_name}{datetime.now().strftime('%y-%m-%d')}")
            schedule = torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2)
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]

            profiler = torch.profiler.profile(
                schedule=schedule,
                activities=activities,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            )
            
        if profiler:
            with profiler:
                train_with_profiler(profiler)
        else:
            train_with_profiler(None)
    except Exception:
        # When any error occurs during the training process, log the error.
        # Note that only the error occured in the rank 0 will be logged into file.
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, args.global_rank, logging.ERROR)
        else:
            print(traceback_info)