import math

import torch
import torch.distributed
from torch.utils.data import DataLoader, IterableDataset, DistributedSampler

from common.lora_modules import *
from common.registry import registry
from dataset_classes import PackingDataset, IterablePackingDataset
from common.utils import DataCollator, PipeLine_Datacollator, print_rank_0

def get_train_eval_args(args, is_train):
    return ('TRAIN' if is_train else 'EVAL',
            args.train_dataset_path if is_train else args.eval_dataset_path, 
            args.max_len if is_train else args.eval_max_len, 
            args.max_src_len if is_train else args.eval_max_src_len,
            args.read_nums if is_train else args.eval_read_nums,
            args.batch_size_per_gpu if is_train else args.eval_batch_size_per_gpu)

def load_dataloder(args, tokenizer, dp_rank, num_dp_ranks, dataset_kwargs, is_train):
    flag, dataset_path, max_len, max_src_len, read_nums, batch_size_per_gpu = get_train_eval_args(args, is_train)
    if dataset_path is None:
        return None
    data_collator = PipeLine_Datacollator(tokenizer) if args.num_pp_stages else DataCollator(tokenizer)
    print_rank_0(f'--->Using dataset class: {args.dataset_class_name}', args.global_rank)
    dataset_class = registry.get_dataset_class(args.dataset_class_name)
    dataset_kwargs = dict(mode=args.mode, 
                        tokenizer=tokenizer,
                        global_rank=args.global_rank,
                        meta_prompt=args.meta_prompt,
                        prefix=args.prefix,
                        postfix=args.postfix,
                        padding=(args.batching_stretegy == 'padding'),
                        dp_rank=dp_rank,
                        num_dp_ranks=num_dp_ranks,
                        encode_single_gene=args.encode_single_gene,
                        shuffle=True,
                        apply_chat_template=False,
                        **dataset_kwargs)
    
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
            update_steps_denominator = num_dp_ranks if is_iterable_dataset else 1
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