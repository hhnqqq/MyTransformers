import os
import math
import torch
import deepspeed
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import common.utils.parallel_states as parallel_states

from gemma.config import *
from gemma.tokenizer import Tokenizer
from common.parser import base_parser, train_parser, ds_parser
from common.dataset import LongRopeDataset
from common.lora import switch_to_lora
from common.optimizer import get_optimizer
from common.registry import registry
from common.utils import init_dist, print_rank_0, read_config, ensure_directory_exists, to_device, get_masks, set_random_seed
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

tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name
tokenizer = registry.get_tokenizer_class(tokenizer_name)(args.tokenizer_path)
print_rank_0('--->loading the model', args.global_rank)
train_model = registry.get_train_model_class(args.model_name)
model = registry.get_model_class(args.model_name)
model = train_model(model=model, args=args, pad_id=tokenizer.pad_id)

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

