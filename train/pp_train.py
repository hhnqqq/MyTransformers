import os
import time
import random

import math
import torch
import deepspeed
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint
from deepspeed.pipe import PipelineModule, TiedLayerSpec, LayerSpec

from train.trainer import Trainer
from model.llama import *
from common.registry import registry
from common.lora import *
from common.dataset import LongRopeDataset
from common.optimizer import get_optimizer
from common.parser import base_parser, train_parser, ds_parser
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from common.utils import print_rank_0, read_config, ensure_directory_exists, set_random_seed, init_dist
from common.utils.params_manager import (
    refresh_config, 
    print_trainable_module_names, 
    enable_trainable_params, 
    disable_untrainable_params
)

class EmbeddingPipelineLayer(torch.nn.Module):
    def __init__(self, model: Transformer, args):
        super().__init__()
        self.args = args
        self.embedder = model.tok_embeddings
        self.weight = self.embedder.weight
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                         args.max_len,
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi,
                                         train_pipeline=True)
        # if args.quant:
        #     self.weight_scaler = self.word_embeddings.weight_scaler

    def forward(self, inputs):
        # [batch_size, input_len, 1]
        input_ids, labels = inputs
        # [batch_size, input_len, hidden_size]
        hidden_states = F.embedding(input_ids, self.weight)
        # Acquire attention mask.
        attention_mask = get_masks(input_ids.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        # Have to set freqs_cis and attention mask trainable, or deepspeed will throw a exception.
        freqs_cis.requires_grad_(True)
        attention_mask.requires_grad_(True)
        return hidden_states, freqs_cis, attention_mask, labels
    
class DecoderPipelineLayer(torch.nn.Module):
    def __init__(self, model: Transformer, layer_idx, args):
        super().__init__()
        self.layer = model.layers[layer_idx]
        self.args = args

    def forward(self, inputs):
        hidden_states, freqs_cis, attention_mask, labels = inputs
        # [batch_size, input_len, hidden_dim]
        if self.args.activation_checkpoint:
            hidden_states = checkpoint(self.layer,  hidden_states, 0, freqs_cis, attention_mask)
        else:
            hidden_states = self.layer(hidden_states, 0, freqs_cis, attention_mask)
        return hidden_states, freqs_cis, attention_mask, labels
    
class FNormPipelineLayer(torch.nn.Module):
    def __init__(self, model: Transformer):
        super().__init__()
        self.final_norm = model.norm
        self.o_proj = model.output

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        # [batch_size, input_len, hidden_dim]
        logits = self.final_norm(hidden_states)
        logits = self.o_proj(logits)
        return logits, labels

class LossPipelineLayer(torch.nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, inputs):
        logits, labels = inputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss
    
def get_model(model, args):
    layers = [LayerSpec(EmbeddingPipelineLayer, model=model, args=args),
              *[LayerSpec(DecoderPipelineLayer, model=model, args=args, layer_idx=idx) for idx in
                range(args.num_layers)],
              LayerSpec(FNormPipelineLayer, model=model),
              LayerSpec(LossPipelineLayer, pad_id=args.pad_id)]
    return layers

def get_masks(seqlen, device, dtype, start_pos=0):
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device),mask]).to(dtype)
        return mask

def data_collator(examples):
    input_ids_list, labels_list = [], []
    for instance in examples:
        input_ids_list.append(instance["input_ids"])
        labels_list.append(instance["labels"])
    return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))
    
args = ds_parser(train_parser(base_parser())).parse_args()
args = registry.get_paths(args)
set_random_seed(args.seed)
device, args = init_dist(args)
tokenizer_name = args.tokenizer_name if args.tokenizer_name is not None else args.model_name
tokenizer = registry.get_tokenizer_class(tokenizer_name)(args.tokenizer_path)
print_rank_0('--->loading the model', args.global_rank)
ModelArgs.vocab_size = tokenizer.n_words
initialize_model_parallel(1)
model = Transformer(ModelArgs)
if args.ckpt_path is not None:
    model.load_weights(args.ckpt_path)
args.head_dim = ModelArgs.dim // ModelArgs.n_heads
args.hidden_size = ModelArgs.dim
args.num_layers = ModelArgs.n_layers
args.pad_id = tokenizer.pad_id

model_pipe = PipelineModule(layers=get_model(model, args), num_stages=args.num_pp_stages, partition_method='uniform')
if args.use_lora or args.use_lora_plus:
    if args.replace_modules is None:
        args.replace_modules = ['wq','wk','wv']
    switch_to_lora(model_pipe, args.replace_modules, rank=4)
    if args.lora_fa:
        enable_trainable_params(model_pipe, ['weight_b'])
    else:
        enable_trainable_params(model_pipe, ['weight_a','weight_b'])
elif args.disable_list is not None:
    disable_untrainable_params(model_pipe, args.disable_list)
elif args.enable_list is not None:
    enable_trainable_params(model_pipe, args.enable_list)
print_trainable_module_names(model_pipe)

if args.fp16:
    model_pipe.to(device).half()
elif args.bf16:
    model_pipe.to(device).bfloat16()

train_dataset = LongRopeDataset(args.data_path, tokenizer, args.max_len, args.max_src_len, args.mode, args.read_nums)
ds_config = read_config(args.ds_config_path, encoding=None)
ds_config = refresh_config(ds_config, args)

g = torch.Generator()
train_dataloader = DataLoader(train_dataset,
                            collate_fn=data_collator,
                            shuffle=True,
                            drop_last=True,
                            batch_size=args.batch_size_per_gpu,
                            generator=g)

assert args.train_iters is not None or args.epochs is not None, 'train_iters and epochs can not be None at the same time'
if args.epochs is not None:
    args.num_update_steps = args.epochs * (math.ceil(len(train_dataloader) / (args.gradient_accumulation_steps)))
else:
    args.num_update_steps = args.train_iters/args.gradient_accumulation_steps
args.num_warmup_steps = int(args.num_update_steps * args.warmup)
ds_config["optimizer"]["scheduler"]["params"]["warmup_num_steps"] = args.num_warmup_steps
print_rank_0("--->TRAIN DATALOADER LENGTH: len(train_dataloader) = {}".format(len(train_dataloader)), args.global_rank)
print_rank_0("--->TRAIN DATASET LENGTH: = {}".format(len(train_dataset)), args.global_rank)
print_rank_0("--->TRAIN BATCH SIZE PER GPU: args.batch_size_per_gpu = {}".format(args.batch_size_per_gpu), args.global_rank)
print_rank_0("--->NUMBER OF UPDATE STEPS: args.num_update_steps = {}".format(args.num_update_steps), args.global_rank)
print_rank_0("--->NUMBER OF WARMUP STEPS: args.num_warmup_steps = {}".format(args.num_warmup_steps), args.global_rank)
# start tranning

train_dataloader = iter(deepspeed.utils.RepeatingLoader(train_dataloader))
optimizer, lr_scheduler = get_optimizer(ds_config, args, model=model_pipe)
engine, optimizer, _, _ = deepspeed.initialize(model=model_pipe, 
                                               optimizer=optimizer, 
                                               lr_scheduler=lr_scheduler,
                                               config=ds_config, 
                                               model_parameters=[p for p in model_pipe.parameters() if p.requires_grad])

if __name__ == '__main__':
    def forward_step(model, data_loader, args):
        return model.train_batch(data_loader), []
    trainer = Trainer(args)
    trainer.train(engine, train_dataloader, forward_step, None, args)