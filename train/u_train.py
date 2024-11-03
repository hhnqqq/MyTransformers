# TODO: 修改使用huggingface训练的代码
import os
import math

import torch
import deepspeed
import torch.distributed
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler

from model import *
from common.lora_modules import *
from train.trainer import Trainer
from common.parser import get_args
from common.registry import registry
from common.optimizer import get_optimizer
import common.utils.parallel_states as parallel_states
from dataset_classes import PackingDataset, IterablePackingDataset, RepeatingLoader
from common.utils.params_manager import refresh_config, set_up_trainable_param, set_up_multimodal_config
from common.utils import (
    DataCollator, PipeLine_Datacollator, load_ckpt_for_train,
    print_rank_0, read_config, set_random_seed, load_ckpt,
    to_device, dict_to_dataclass, reduce_tensor, set_default_tensor_type)

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

def load_huggingface_model_config(args):
    model_name_or_path = args.model_name_or_path
    if os.path.exists(model_name_or_path):
        config_path = os.path.join(model_name_or_path, 'config.json')
        # Expecially, write lora layers config in the config file is needed.
        model_config = read_config(config_path)
        # Convert dict config to dataclass.
        model_config = dict_to_dataclass(model_config)
    else:
        model_config = None
    return model_config        

def load_huggingface_model(args):
    # Train with huggingface pretrained model. Only support data parallel training.
    return_dataset_kwargs = {}
    assert args.num_pp_stages is None
    print_rank_0(f'--->Using tokenizer and model from huggingface with path: {args.model_name_or_path}', args.global_rank)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    # Load trainable params if needed.

    if args.partial_ckpt_path:
        load_ckpt(model=model, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
    model_config = load_huggingface_model_config(args)
    return model, tokenizer, model_config, return_dataset_kwargs


def load_local_model(args):
    # Train with local defined model.
    return_dataset_kwargs = {}
    tokenizer = registry.get_tokenizer_class(args.tokenizer_name)(args.tokenizer_path)
    print_rank_0(f'--->Using tokenizer: {args.tokenizer_name} with path: {args.tokenizer_path}', args.global_rank)
    config_type = '_'.join([args.model_name, args.variant])
    model_config = registry.get_model_config_class(config_type)()
    if args.multimodal:
        set_up_multimodal_config(model_config, args)
        return_dataset_kwargs['multimodal_k_tokens'] = args.multimodal_k_tokens
    if 'concat' in args.dataset_class_name and args.dataset_weights is not None:
        return_dataset_kwargs['weights'] = [int(i) for i in args.dataset_weights.split('_')]
    print_rank_0(f'--->Using model config: {config_type}', args.global_rank)
    model_config.vocab_size = tokenizer.n_words
    with set_default_tensor_type(args.default_dtype):
        model = registry.get_model_class(args.model_name)(model_config)
    print_rank_0(f'--->Using model: {args.model_name}, and loading its trainning variant', args.global_rank)

    # Load checkpoint if checkpoint path is provieded.
    if args.ckpt_path is not None and args.from_pretrained:
        if args.model_name == 'gemma':
            _, optimizer_sd, lr_scheduler_sd = load_ckpt_for_train(model=model, ckpt_path=args.ckpt_path, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
        else:
            _, optimizer_sd, lr_scheduler_sd = load_ckpt_for_train(model=model.model, ckpt_path=args.ckpt_path, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
        print_rank_0(f'--->Using pretrained checkpoint at {args.ckpt_path}', args.global_rank)
        model_config.optimizer_sd = optimizer_sd
        model_config.lr_scheduler_sd = lr_scheduler_sd

        if args.multimodal and hasattr(model, 'multimodal_model'):
            load_ckpt(model=model.multimodal_model, ckpt_path=args.multimodal_model_ckpt_path, rank=args.global_rank)
            print_rank_0(f'--->Using pretrained multimodal model checkpoint at {args.multimodal_model_ckpt_path}', args.global_rank)
            multimodal_tokenizer = registry.get_tokenizer_class(args.multimodal_tokenizer_name)(model_config.multimodal_model_config.tokenizer)
            return_dataset_kwargs['multimodal_tokenizer'] = multimodal_tokenizer
    else:
        print_rank_0('--->Not using pretrained checkpoint to start traning.', args.global_rank)

    # Load config from model config to argument parser namespace.
    args.head_dim = model_config.head_dim
    args.head_num = model_config.n_heads
    args.hidden_size = model_config.dim
    args.num_layers = model_config.n_layers
    args.rope_theta = model_config.rope_theta if args.rope_theta is None else args.rope_theta
    args.pad_id = tokenizer.pad_id

    # Load model to training dtype.
    if args.fp16:
        model.half().to(args.device)
    elif args.bf16:
        model.bfloat16().to(args.device)

    # Convert model to trainable model for given training type.
    if args.num_pp_stages:
        pipe_model_cls = registry.get_pipeline_model_class(args.model_name)
        model = pipe_model_cls(model, args)
    else:
        train_model_cls = registry.get_train_model_class(args.model_name)
        model = train_model_cls(model, args)
    return model, tokenizer, model_config, return_dataset_kwargs

def get_train_eval_args(args, is_train):
    return ('TRAIN' if is_train else 'EVAL',
            args.train_dataset_path if is_train else args.eval_dataset_path, 
            args.max_len if is_train else args.eval_max_len, 
            args.max_src_len if is_train else args.eval_max_src_len,
            args.read_nums if is_train else args.eval_read_nums,
            args.batch_size_per_gpu if is_train else args.eval_batch_size_per_gpu)

def load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, is_train):
    flag, dataset_path, max_len, max_src_len, read_nums, batch_size_per_gpu = get_train_eval_args(args, is_train)
    if dataset_path is None:
        return None
    data_collator = PipeLine_Datacollator(tokenizer) if args.num_pp_stages else DataCollator(tokenizer)
    print_rank_0(f'--->Using dataset class: {args.dataset_class_name}', args.global_rank)
    dataset_class = registry.get_dataset_class(args.dataset_class_name)
    dataset_kwargs = dict(mode=args.mode, 
                        tokenizer = tokenizer,
                        global_rank=args.global_rank,
                        meta_prompt=args.meta_prompt,
                        prefix=args.prefix,
                        postfix=args.postfix,
                        padding=(args.batching_stretegy == 'padding'),
                        dp_rank=dp_rank,
                        num_dp_ranks=num_dp_rank,
                        encode_single_gene=args.encode_single_gene,
                        shuffle=True,
                        **return_dataset_kwargs)
    
    dataset = dataset_class(
        dataset_path, max_len=max_len, max_src_len=max_src_len,
        read_nums=read_nums, **dataset_kwargs
    )
    is_iterable_dataset = isinstance(dataset, IterableDataset)
    dataset_sampler = None if is_iterable_dataset else DistributedSampler(dataset, num_dp_ranks, dp_rank)
    if args.batching_stretegy == 'packing':
        if is_iterable_dataset:
            dataset = IterablePackingDataset(dataset, chunk_size=max_len)
        else:
            dataset = PackingDataset(dataset, chunk_size=max_len)

    dataloader = DataLoader(dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            sampler=dataset_sampler,
                            batch_size=batch_size_per_gpu,
                            generator=torch.Generator())
    
    msgs = [
        f"{flag} DATALOADER LENGTH: {len(dataloader)}",
        f"{flag} DATASET LENGTH: {len(dataset)}",
        f"{flag} BATCH SIZE PER GPU: {batch_size_per_gpu}"
        ]
    if is_train:
        assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
        if args.epochs is not None:
            update_steps_denominator = num_dp_rank if is_iterable_dataset else 1
            micro_update_steps_one_epoch = math.ceil(len(dataloader) / update_steps_denominator)
            args.num_micro_update_steps = args.epochs * (math.ceil(micro_update_steps_one_epoch))
        else:
            args.num_micro_update_steps = args.train_iters
        args.num_global_update_steps = math.ceil(args.num_micro_update_steps / args.gradient_accumulation_steps)
        args.num_warmup_steps = int(args.num_global_update_steps * args.warmup) + 1
        msgs.extend([f"NUMBER OF MICRO UPDATE STEPS: {args.num_micro_update_steps}",
            f"NUMBER OF GLOBAL UPDATE STEPS: {args.num_global_update_steps}",
            f"NUMBER OF WARMUP STEPS: {args.num_warmup_steps}",
            f"Base learning rate is {args.lr}"
            ])
        
    for msg in msgs:
        print_rank_0(f"--->{msg}", args.global_rank)
    return dataloader

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
train_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_rank, True)
eval_dataloader = load_dataloder(args, tokenizer, dp_rank, num_dp_rank, False)

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