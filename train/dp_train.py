import math
import deepspeed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import common.utils.parallel_states as parallel_states

from model import *
from train.trainer import Trainer
from common.registry import registry
from common.lora import switch_to_lora
from common.optimizer import get_optimizer
from common.dataset import LongRopeDataset
from common.parser import base_parser, train_parser, ds_parser
from common.utils import init_dist, print_rank_0, read_config, set_random_seed, to_device, DataCollator
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
print_rank_0(f'--->registry contains {registry.list_all()}')
tokenizer = registry.get_tokenizer_class(args.tokenizer_name)(args.tokenizer_path)
print_rank_0(f'--->using tokenizer: {args.tokenizer_name} with path: {args.tokenizer_path}', args.global_rank)
config_type = '_'.join([args.model_name, args.variant])
model_config = registry.get_model_config_class(config_type)()
print_rank_0(f'--->using model config: {config_type}', args.global_rank)
model_config.vocab_size = tokenizer.n_words
model = registry.get_model_class(args.model_name)(model_config, is_train=True)
print_rank_0(f'--->using model: {args.model_name} and loading its dp train variant', args.global_rank)
train_model = registry.get_train_model_class(args.model_name)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)

args.head_dim = model_config.head_dim
args.head_num = model_config.num_attention_heads
args.hidden_size = model_config.hidden_size
args.num_layers = model_config.num_hidden_layers
model = train_model(model=model, args=args, pad_id=tokenizer.pad_id)


if args.use_lora or args.use_lora_plus:
    if args.replace_modules is None:
        args.replace_modules = model_config.lora_layers
    switch_to_lora(model, args.replace_modules, rank=4)
    if args.lora_fa:
        enable_trainable_params(model, ['weight_b'])
    else:
        enable_trainable_params(model, ['weight_a','weight_b'])
elif args.disable_list is not None:
    disable_untrainable_params(model, args.disable_list[0].split(','))
elif args.enable_list is not None:
    enable_trainable_params(model, args.enable_list)

print_trainable_module_names(model)

if args.fp16:
    model.to(device).half()
elif args.bf16:
    model.to(device).bfloat16()

train_dataset = LongRopeDataset(args.dataset_path, 
                                tokenizer, 
                                args.max_len, 
                                args.max_src_len, 
                                args.mode, 
                                args.read_nums,
                                args.global_rank)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

if args.num_sp_stages is not None:
    train_sampler = SequentialSampler(train_dataset)
elif args.local_rank == -1 :
    train_sampler = RandomSampler(train_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
data_collator = DataCollator(tokenizer)
train_dataloader = iter(deepspeed.utils.RepeatingLoader(DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler,
                                batch_size=args.batch_size_per_gpu)))

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
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


    def forward_step(model, data_loader, args, step):
        batch = next(data_loader)
        batch = to_device(batch, device)
        loss = model(**batch)
        return loss, []
    
    def backward_step(model, optimizer, loss):
        model.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        model.step()
        return model
    
    trainer = Trainer(args, writer)
    try:
        trainer.train(model=model, 
                      data_loader=train_dataloader, 
                      optimizer=None, 
                      forward_step=forward_step, 
                      backward_step=backward_step, 
                      log_loss=True)
    except:
        # When any error occur during the training process, log the error
        traceback_info = traceback.format_exc()
        print_rank_0(traceback_info, args.global_rank, logging.ERROR)