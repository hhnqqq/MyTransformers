# TODO: 修改使用huggingface训练的代码
import os

import torch
import deepspeed
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
from train.load_model import load_huggingface_model, load_local_model
from dataset_classes import RepeatingLoader
from common.utils.params_manager import refresh_config, set_up_trainable_param
from common.utils import print_rank_0, read_config, set_random_seed, to_device, reduce_tensor

args = get_args()
# If args.test_code, the log file and tb writer will not be created.
if args.test_code:
    os.environ['NO_LOG_FILE'] = 'true'
args = registry.get_paths(args)
set_random_seed(args.seed)

print_rank_0(f'--->Data parallel world size: {parallel_states.get_data_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Sequence parallel world size: {parallel_states.get_sequence_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->Pipeline parallel world size: {parallel_states.get_pipeline_model_parallel_world_size()}', args.global_rank)
print_rank_0(f'--->registry contains {registry.list_all()}', args.global_rank)

print_rank_0('--->loading the model', args.global_rank)
if args.huggingface:
    model, tokenizer, model_config, return_dataset_kwargs = load_huggingface_model(args)
else:
    model, tokenizer, model_config, return_dataset_kwargs = load_local_model(args)

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

engine, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, 
                                               optimizer=optimizer, 
                                               lr_scheduler=lr_scheduler,
                                               config=ds_config, 
                                               model_parameters=[p for p in model.parameters() if p.requires_grad],
                                               mpu=None if args.num_pp_stages else parallel_states)


if __name__ == '__main__':

    # import wandb
    import logging
    import traceback
    import torch.profiler as profiler

    from argparse import Namespace
    from torch.profiler import ProfilerActivity, record_function
    from deepspeed.runtime.pipe.engine import PipelineEngine, DeepSpeedEngine

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

    def forward_step_pipeline(model: PipelineEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
        with record_function("forward_backward_path"):
            return model.train_batch(data_loader), []

    def eval_step_pipeline(model: PipelineEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
        with record_function("eval_path"):
            return model.eval_batch(data_loader).item()

    def forward_step_deepspeed(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
        with torch.profiler.record_function("get_data"):
            batch = next(data_loader)
            batch = to_device(batch, args.device)

        with torch.profiler.record_function("forward_path"):
            if args.huggingface:
                loss = model(**batch).loss
                metric = {}
            else:
                loss, metric = model(**batch)

            if args.all_reduce_loss:
                # Reduce loss for average loss print, not for backpropagation.
                # DeepSpeed uses on-chip loss for backpropagation and all-reduces gradients afterwards.
                loss_reduced = reduce_tensor(loss, args.world_size)
                metric['loss_reduced'] = loss_reduced
                del loss_reduced
                
            return loss, metric
        
    def backward_step_deepspeed(model: DeepSpeedEngine, optimizer, loss):
        with record_function("backward_path"):
            model.backward(loss)
            if not args.not_clip_grad_norm:
                # this should be disabled if deepspeep already config cliping.
                # deepspeed/runtime/engine.py ##line 2068
                torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            args.clip_grad_max_norm, 
                                            args.clip_grad_norm_type)
            # deepspeed/runtime/engine.py ##line 2134
            # Only update model when self.is_gradient_accumulation_boundary()
            model.step()
            return model

    def eval_step_deepspeed(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
        with record_function("eval_path"):
            batch = next(data_loader)
            batch = to_device(batch, args.device)
            with torch.no_grad():
                loss, metric = model(**batch)
                # TODO: all reduce metrics
            return loss.item(), metric

    def task_print_pipeline(all_metric, args):
        return ''

    def task_print_deepspeed(all_metric, args):
        # return the on-chip accuracy
        acc_count = sum([sub_dict.get("accuracy", 0) for sub_dict in all_metric])
        mcc_count = sum([sub_dict.get("mcc", 0) for sub_dict in all_metric])
        return f' acc:{(acc_count/args.show_loss_step) * 100}, mcc:{(mcc_count/args.show_loss_step) * 100}'
    
    writer = get_writer(args)
    if args.num_pp_stages:
        forward_step = forward_step_pipeline
        eval_step = eval_step_pipeline
        backward_step = None
        task_print = task_print_pipeline
    else:
        forward_step = forward_step_deepspeed
        eval_step = eval_step_deepspeed
        backward_step = backward_step_deepspeed
        task_print = task_print_deepspeed

    trainer = Trainer(args, writer)
    trainer.register_task_print(task_print)

    try:
        if args.profile_log_dir:
            log_dir = os.path.join(args.profile_log_dir, args.experiment_name + datetime.now().strftime('%y-%m-%d'))
            with profiler.profile(
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2),
            activities=[ProfilerActivity.CPU, 
                        ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
            ) as prof:
                trainer.train(model=engine,
                            train_data_loader=RepeatingLoader(train_dataloader),
                            eval_data_loader=RepeatingLoader(eval_dataloader),
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            forward_step=forward_step,
                            backward_step=backward_step,
                            eval_step=eval_step,
                            profiler=prof,
                            log_loss=True)
        else:
            trainer.train(model=engine,
                        train_data_loader=RepeatingLoader(train_dataloader),
                        eval_data_loader=RepeatingLoader(eval_dataloader),
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        forward_step=forward_step,
                        backward_step=backward_step,
                        eval_step=eval_step,
                        profiler=None,
                        log_loss=True)
    except Exception as e:
        # When any error occurs during the training process, log the error.
        # Note that only the error occured in the rank 0 will be logged into file.
        traceback_info = traceback.format_exc()
        if args.global_rank == 0:
            print_rank_0(traceback_info, args.global_rank, logging.ERROR)
        else:
            print(traceback_info)