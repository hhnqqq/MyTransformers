import math
import deepspeed
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

import common.utils.parallel_states as parallel_states
from gemma.config import *
from train.trainer import Trainer
from common.parser import base_parser, train_parser, ds_parser
from common.dataset import LongRopeDataset
from common.lora import switch_to_lora
from common.optimizer import get_optimizer
from common.registry import registry
from common.utils import init_dist, print_rank_0, read_config, set_random_seed, DataCollator
from common.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)

args = ds_parser(train_parser(base_parser())).parse_args()
args = registry.get_paths(args)
set_random_seed(args.seed)
device, args = init_dist(args)

print_rank_0('--->loading the model', args.global_rank)
tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name
tokenizer = registry.get_tokenizer_class(tokenizer_name)(args.tokenizer_path)
model_config = registry.get_model_config_class('_'.join([args.model_name, args.variant]))
model = registry.get_model_class(args.model_name)(model_config, is_train=True)
train_model = registry.get_train_model_class(args.model_name)
model = train_model(model=model, args=args, pad_id=tokenizer.pad_id)
args.head_dim = model_config.head_dim
args.head_num = model_config.num_attention_heads
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers

init_dist()

if args.use_lora or args.use_lora_plus:
    switch_to_lora(model, args.replace_modules, rank=args.lora_rank, use_dora=args.use_dora)
    if args.lora_fa:
        enable_trainable_params(model, ['weight_b'])
    else:
        enable_trainable_params(model, ['weight_a', 'weight_b'])
elif args.disable_list or args.enable_list:
    param_list = args.disable_list if args.disable_list is not None else args.enable_list
    disable_untrainable_params(model, param_list) if args.disable_list else enable_trainable_params(model, param_list)

if args.fp16:
    model.to(device).half()
elif args.bf16:
    model.to(device).bfloat16()

train_dataset = LongRopeDataset(args.data_path, 
                                tokenizer, 
                                args.max_len, 
                                args.max_src_len, 
                                args.mode, 
                                args.read_nums,
                                args.global_rank)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

if args.local_rank == -1 or args.num_sp_stages is not None:
    train_sampler = RandomSampler(train_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
data_collator = DataCollator(tokenizer)
train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                batch_size=args.batch_size_per_gpu)

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    # TODO: 修正sp与dp时的不同
    args.num_update_steps = args.epochs * (math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps)))
else:
    args.num_update_steps = args.train_iters/args.gradient_accumulation_steps
args.num_warmup_steps = int(args.num_update_steps * args.warmup) + 1
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
print_rank_0("--->TRAIN DATALOADER LENGTH: len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
print_rank_0("--->TRAIN DATASET LENGTH: = {}".format(len(train_dataset)), args.global_rank)
print_rank_0("--->TRAIN BATCH SIZE PER GPU: args.batch_size_per_gpu = {}".format(args.batch_size_per_gpu), args.global_rank)
print_rank_0("--->NUMBER OF UPDATE STEPS: args.num_update_steps = {}".format(args.num_update_steps), args.global_rank)
print_rank_0("--->NUMBER OF WARMUP STEPS: args.num_warmup_steps = {}".format(args.num_warmup_steps), args.global_rank)

optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model)
model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, 
                                                         optimizer=optimizer,
                                                         lr_scheduler=lr_scheduler,
                                                         config=ds_config,
                                                         model_parameters=[p for p in model.parameters() if p.requires_grad],
                                                         mpu=parallel_states)
print_trainable_module_names(model)

if __name__ == '__main__':
    import logging
    import traceback
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            from tensorboard import SummaryWriter
        writer = SummaryWriter()
    else:
        writer = None

    def forward_step(model, data_loader, _):
        return model.train_batch(data_loader), []
    trainer = Trainer(args, writer)
    try:
        trainer.train(model=model, 
                      data_loader=train_dataloader, 
                      optimizer=None, 
                      forward_step=forward_step, 
                      backward_step=None, 
                      log_loss=True)
    except:
        # When any error occur during the training process, log the error
        traceback_info = traceback.format_exc()
        print_rank_0(traceback_info, args.global_rank, logging.ERROR)