import torch
from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import DeepSpeedEngine

from dataset_classes import RepeatingLoader
from common.lora_modules.relora import optimizer_reset
from common.lora_modules.lora import LinearWithLoRA, find_lora_names
from common.utils import to_device, reduce_tensor, print_rank_0

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
    
def backward_step_deepspeed(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        model.backward(loss)
        # deepspeed/runtime/engine.py ##line 2134
        # Only update model when self.is_gradient_accumulation_boundary()
        model.step()

    return model

def backward_step_deepspeed_relora(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        model.backward(loss)
        model.step()

    if args.relora_steps and (step / args.gradient_accumulation_steps) % args.relora_steps == 0:
        for module in model.modules():
            if isinstance(module, LinearWithLoRA):
                module.merge_and_reset()
        if args.relora_reset_optimizer:
            optimizer_reset(
                optimizer,
                lr_scheduler,
                reset_params=[p for n, p in model.named_parameters() if p.requires_grad and find_lora_names(n)],
                reset_optimizer_on_relora=args.relora_reset_optimizer,
                optimizer_random_pruning=args.relora_optimizer_random_pruning,
                optimizer_magnitude_pruning=args.relora_optimizer_magnitude_pruning,
                args=args
            )
        return model
    
def reduce_gradients(model, world_size):
    for param in model.parameters():
        if param.requires_grad:
            param.grad.data = reduce_tensor(param.grad.data, world_size)

def backward_step_deepspeed_stage0(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    loss = loss / args.gradient_accumulation_steps
    loss.backward()
    if step % args.gradient_accumulation_steps == 0:
        reduce_gradients(model, args.world_size)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
    elif step == args.num_micro_update_steps:
        reduce_gradients(model, args.world_size)
        optimizer.step()
        optimizer.zero_grad()

def eval_step_deepspeed(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with record_function("eval_path"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)
        with torch.no_grad():
            loss, metric = model(**batch)
            # TODO: all reduce metrics
        return loss.item(), metric
    
def task_print_ntp(all_metric, args):
    return ""

def task_print_bio(all_metric, args):
    # return the on-chip accuracy
    acc_count = sum([sub_dict.get("accuracy", 0) for sub_dict in all_metric])
    mcc_count = sum([sub_dict.get("mcc", 0) for sub_dict in all_metric])
    return f' acc:{(acc_count/args.show_loss_step) * 100}, mcc:{(mcc_count/args.show_loss_step) * 100}'