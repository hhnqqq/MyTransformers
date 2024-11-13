import torch
import deepspeed
from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import DeepSpeedEngine

from contextlib import contextmanager
from dataset_classes import RepeatingLoader
from common.lora_modules.relora import optimizer_reset
from common.lora_modules import LinearWithLoRA, LinearWithPLoRA, find_lora_names
from common.lora_modules.delta_lora import LinearWithDeltaLoRA
from common.lora_modules.adalora import update_and_allocate
from common.utils import to_device, reduce_tensor

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

    if (step / args.gradient_accumulation_steps) % args.relora_steps == 0:
        for module in model.modules():
            if isinstance(module, LinearWithLoRA):
                module.merge_and_reset()
        if args.relora_reset_optimizer:
            optimizer_reset(
                optimizer,
                lr_scheduler,
                relora_auto_warmup_steps = args.relora_auto_warmup_steps,
                relora_auto_warmup_ratio = args.relora_auto_warmup_ratio,
                reset_params=[p for n, p in model.named_parameters() if p.requires_grad and find_lora_names(n)],
                reset_optimizer_on_relora=args.relora_reset_optimizer,
                optimizer_random_pruning=args.relora_optimizer_random_pruning,
                optimizer_magnitude_pruning=args.relora_optimizer_magnitude_pruning,
                args=args
            )
    return model

def backward_step_deepspeed_plora(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        model.backward(loss)
        model.step()

    if step % args.gradient_accumulation_steps == 0:
        for module in model.modules():
            if isinstance(module, LinearWithPLoRA):
                module.merge_and_reset_with_momentum()
    return model

def backward_step_deepspeed_deltalora(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        model.backward(loss)
        model.step()

    if args.delta_lora_start_steps and (step / args.gradient_accumulation_steps) > args.delta_lora_start_steps:
        for module in model.modules():
            if isinstance(module, LinearWithDeltaLoRA):
                module.update_pretrained_weight()
        return model

def adalora_step(args, loss, model):
    # reference: https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/model.py#L238
    orth_reg_weight = args.orth_reg_weight
    if orth_reg_weight <= 0:
        raise ValueError("orth_reg_weight should be greater than 0. ")

    regu_loss = 0
    num_param = 0
    for n, p in model.named_parameters():
        if "weight_a" in n or "weight_b" in n:
            if p.shape == torch.Size([0]):
                with gather_params_ctx(p, fwd_module=model):
                    para_cov = p @ p.T if "weight_a" in n else p.T @ p
            else:
                para_cov = p @ p.T if "weight_a" in n else p.T @ p
            I = torch.eye(*para_cov.size(), out=torch.empty_like(para_cov))  # noqa: E741
            I.requires_grad = False
            num_param += 1
            regu_loss += torch.norm(para_cov - I, p="fro")
    if num_param > 0:
        regu_loss = regu_loss / num_param
    else:
        regu_loss = 0
    loss += orth_reg_weight * regu_loss

    return loss

@contextmanager
def gather_params_ctx(param, modifier_rank: int = 0, fwd_module: torch.nn.Module = None):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""

    with deepspeed.zero.GatheredParameters(param, modifier_rank=modifier_rank, fwd_module=fwd_module):
        yield
    return

def forward_step_deepspeed_adalora(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with torch.profiler.record_function("get_data"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)

    with torch.profiler.record_function("forward_path"):
        if args.huggingface:
            loss = model(**batch).loss
            metric = {}
        else:
            loss, metric = model(**batch)


        loss = adalora_step(args, loss, model)

        if args.all_reduce_loss:
            # Reduce loss for average loss print, not for backpropagation.
            # DeepSpeed uses on-chip loss for backpropagation and all-reduces gradients afterwards.
            loss_reduced = reduce_tensor(loss, args.world_size)
            metric['loss_reduced'] = loss_reduced
            del loss_reduced
            
        return loss, metric
    
def backward_step_deepspeed_adalora(model: DeepSpeedEngine, optimizer, loss, lr_scheduler, args, step):
    with record_function("backward_path"):
        model.backward(loss)

        update_flag = False

        # here we need to update and allocate the rank of adalora model
        if model.is_gradient_accumulation_boundary():
            saved_gradients = {name: deepspeed.utils.safe_get_full_grad(param) for name, param in model.named_parameters() if deepspeed.utils.safe_get_full_grad(param) is not None}
            update_flag = True

        # deepspeed/runtime/engine.py ##line 2134
        # Only update model when self.is_gradient_accumulation_boundary()
        model.step()

        if update_flag:
            update_and_allocate(model, step, saved_gradients)
            update_flag = False

        return model

def eval_step_deepspeed_adalora(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with record_function("eval_path"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)
        with torch.no_grad():
            loss, metric = model(**batch)
            loss = adalora_step(args, loss, model)
            # TODO: all reduce metrics

        return loss.item(), metric

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
