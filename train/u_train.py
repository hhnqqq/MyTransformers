import os
import torch
import torch.distributed
from datetime import datetime

from model import *
from common.lora_modules import *
from train.trainer import Trainer
from common.parser import get_args
from common.registry import registry
from common.optimizer import get_optimizer
import common.utils.parallel_states as parallel_states
from train.load_data import load_dataloder
from train.load_model import load_model
from dataset_classes import RepeatingLoader
from common.utils.params_manager import refresh_config, set_up_trainable_param
from common.utils import print_rank_0, read_config, set_random_seed, init_distributed_model

args = get_args()
args = registry.get_paths(args)
set_random_seed(args.seed)

print_rank_0(f'--->Data parallel world size: {parallel_states.get_data_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Sequence parallel world size: {parallel_states.get_sequence_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Pipeline parallel world size: {parallel_states.get_pipeline_model_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Registry contains {registry.list_all()}', args.global_rank)

print_rank_0('--->Loading the model', args.global_rank)
model, tokenizer, model_config, return_dataset_kwargs = load_model(args)

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
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps

# start tranning

# Run this befor set up trainable parameters.
if args.use_lora_ga:
    lora_ga_reinit(model=model,
                   dataloader=train_dataloader,
                   args=args,
                   iters=args.lora_ga_n_steps)
# set up trainable before acquiring optimizer.
set_up_trainable_param(model, args)

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

    # import wandb
    import logging
    import traceback
    import torch.profiler as profiler

    from torch.profiler import ProfilerActivity
    from train.pp_train import *
    from train.dp_train import *

    def get_writer(args):
        # if args.wandb:
        #     wandb.init(project=args.experiment_name,
        #                config=args,
        #                entity='Shanghai AI Lab',
        #                sync_tensorboard=True)
        if args.tensorboard and not args.test_code:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                from tensorboard import SummaryWriter
            log_dir = os.path.join(args.tb_log_dir, args.experiment_name + datetime.now().strftime('%y-%m-%d_%H-%M'))
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
        if args.disable_zero_optimizer:
            backward_step = backward_step_deepspeed_stage0
        elif args.relora_steps:
            backward_step = backward_step_deepspeed_relora
        elif args.use_plora:
            backward_step = backward_step_deepspeed_plora
        elif args.use_delta_lora:
            backward_step = backward_step_deepspeed_deltalora
        else:
            backward_step = backward_step_deepspeed
        task_print = task_print_ntp

    trainer = Trainer(args, writer)
    trainer.register_task_print(task_print)

    def train_with_profiler(profiler):
        trainer.train(
            model=engine,
            train_data_loader=RepeatingLoader(train_dataloader),
            eval_data_loader=RepeatingLoader(eval_dataloader),
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
    except Exception as e:
        # When any error occurs during the training process, log the error.
        # Note that only the error occured in the rank 0 will be logged into file.
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, args.global_rank, logging.ERROR)
        else:
            print(traceback_info)