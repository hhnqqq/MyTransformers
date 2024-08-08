# 备份文件
import os
import math
import torch
import deepspeed

from torch.utils.data import DataLoader
from deepspeed.utils import RepeatingLoader

from train.trainer import Trainer
from model import *
from common.registry import registry
from common.lora import *
from common.dataset import IterableDataset
from common.optimizer import get_optimizer
from common.parser import base_parser, train_parser, ds_parser
from common.utils import print_rank_0, read_config, set_random_seed, init_dist, to_device
from common.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)

def data_collator(examples):
    input_ids_list, labels_list = [], []
    for instance in examples:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["labels"])
    return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))

args = ds_parser(train_parser(base_parser())).parse_args()
if args.test_code:
    os.environ['NO_LOG_FILE'] = 'true'
args = registry.get_paths(args)
set_random_seed(args.seed)
device, args = init_dist(args)



print_rank_0('--->loading the model', args.global_rank)
print_rank_0(f'--->registry contains {registry.list_all()}', args.global_rank)
tokenizer = registry.get_tokenizer_class(args.tokenizer_name)(args.tokenizer_path)
print_rank_0(f'--->using tokenizer: {args.tokenizer_name} with path: {args.tokenizer_path}', args.global_rank)
config_type = '_'.join([args.model_name, args.variant])
model_config = registry.get_model_config_class(config_type)()
print_rank_0(f'--->using model config: {config_type}', args.global_rank)
model_config.vocab_size = tokenizer.n_words
model = registry.get_model_class(args.model_name)(model_config)
print_rank_0(f'--->using model: {args.model_name}, and loading its pipeline variant', args.global_rank)
model_pipe_cls = registry.get_pipeline_model_class(args.model_name)

if args.ckpt_path is not None and args.from_pretrained:
    model.load_weights(args.ckpt_path)
    print_rank_0(f'--->Using pretrained checkpoint at {args.ckpt_path}', args.global_rank)
else:
    print_rank_0('--->Not using pretrained checkpoint to start traning.', args.global_rank)
# resize_embedding(model, getattr(tokenizer, "dna_vocab_size", 0))
args.head_dim = model_config.head_dim
args.head_num = model_config.num_attention_heads
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers
args.rope_theta = model_config.rope_theta if args.rope_theta is None else args.rope_theta
args.pad_id = tokenizer.pad_id

model_pipe = model_pipe_cls(model, args)

if args.use_lora or args.use_lora_plus:
    if args.replace_modules is None:
        args.replace_modules = model_config.lora_layers
    switch_to_lora(model_pipe, args.replace_modules, rank=args.lora_rank)
    if args.lora_fa:
        lora_weight = ['weight_b']
    else:
        lora_weight = ['weight_a','weight_b']
    enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
if args.enable_list is not None:
    enable_trainable_params(model_pipe, args.enable_list)
elif args.disable_list is not None:
    disable_untrainable_params(model_pipe, args.disable_list)
print_trainable_module_names(model_pipe)

if args.fp16:
    model_pipe.to(device).half()
elif args.bf16:
    model_pipe.to(device).bfloat16()

dataset_args = dict(mode=args.mode, 
                    global_rank=args.global_rank,
                    meta_prompt=args.meta_prompt,
                    prefix=args.prefix,
                    postfix=args.postfix)

train_dataset = IterableDataset(args.train_dataset_path, 
                                tokenizer, 
                                max_len=args.max_len, 
                                max_src_len=args.max_src_len,
                                read_nums=args.read_nums,
                                shuffle=True,
                                **dataset_args)
g = torch.Generator()
train_dataloader = DataLoader(train_dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            batch_size=args.batch_size_per_gpu,
                            generator=g)
print_rank_0(f"--->TRAIN DATALOADER LENGTH: len(train_dataloader) = {len(train_dataloader)}", args.global_rank)
print_rank_0(f"--->TRAIN DATASET LENGTH: = {len(train_dataset)}", args.global_rank)
print_rank_0(f"--->TRAIN BATCH SIZE PER GPU: args.batch_size_per_gpu = {args.batch_size_per_gpu}", args.global_rank)
train_dataloader_iter = iter(RepeatingLoader(train_dataloader))

if args.eval_dataset_path is not None:
    eval_dataset = IterableDataset(args.eval_dataset_path, 
                                    tokenizer, 
                                    max_len=args.eval_max_len,
                                    max_src_len=args.eval_max_src_len,
                                    read_nums=args.eval_read_nums,
                                    **dataset_args)
    eval_dataloader = DataLoader(eval_dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            batch_size=args.eval_batch_size_per_gpu,
                            generator=g)
    print_rank_0(f"--->EVAL DATALOADER LENGTH: len(eval_dataloader) = {len(eval_dataloader)}", args.global_rank)
    print_rank_0(f"--->EVAL DATASET LENGTH: = {len(eval_dataset)}", args.global_rank)
    print_rank_0(f"--->EVAL BATCH SIZE PER GPU: args.eval_batch_size_per_gpu = {args.eval_batch_size_per_gpu}", args.global_rank)
    eval_dataloader_iter = iter(RepeatingLoader(train_dataloader))
else:
    eval_dataloader_iter = None

ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    args.num_update_steps = args.epochs * (math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps)))
else:
    args.num_update_steps = args.train_iters/args.gradient_accumulation_steps
args.num_warmup_steps = int(args.num_update_steps * args.warmup) + 1
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
print_rank_0(f"--->NUMBER OF UPDATE STEPS: args.num_update_steps = {args.num_update_steps}", args.global_rank)
print_rank_0(f"--->NUMBER OF WARMUP STEPS: args.num_warmup_steps = {args.num_warmup_steps}", args.global_rank)
print_rank_0(f"--->Base learning rate is {args.lr}", args.global_rank)
# start tranning

optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model_pipe)
engine, optimizer, _, _ = deepspeed.initialize(model=model_pipe, 
                                               optimizer=optimizer, 
                                               lr_scheduler=lr_scheduler,
                                               config=ds_config, 
                                               model_parameters=[p for p in model_pipe.parameters() if p.requires_grad])

if __name__ == '__main__':
    import logging
    import traceback

    from deepspeed.runtime.pipe.engine import PipelineEngine
    from argparse import Namespace

    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboard import SummaryWriter
        writer = SummaryWriter()
    else:
        writer = None

    def forward_step(model:PipelineEngine, 
                    data_loader:RepeatingLoader, 
                    args:Namespace, 
                    step:int):
        return model.train_batch(data_loader), []
    
    def eval_step(model:PipelineEngine, 
                data_loader:RepeatingLoader, 
                args:Namespace, 
                step:int):
        return model.eval_batch(data_loader).item()
    
        
    trainer = Trainer(args, writer)
    args.gradient_accumulation_steps=1 # For correctly print info
    try:
        trainer.train(model=engine, 
                      train_data_loader=train_dataloader_iter, 
                      eval_data_loader=eval_dataloader_iter,
                      optimizer=None, 
                      forward_step=forward_step, 
                      backward_step=None, 
                      eval_step=eval_step,
                      log_loss=True)
    except:
        # When any error occur during the training process, log the error
        traceback_info = traceback.format_exc()
        print_rank_0(traceback_info, args.global_rank, logging.ERROR)