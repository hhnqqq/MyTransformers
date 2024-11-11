import torch
from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import DeepSpeedEngine

from common.utils import to_device, reduce_tensor
from dataset_classes import RepeatingLoader

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
    
def backward_step_deepspeed(model: DeepSpeedEngine, optimizer, loss, step: int):
    with record_function("backward_path"):
        model.backward(loss)
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
    
def task_print_deepspeed(all_metric, args):
    # return the on-chip accuracy
    acc_count = sum([sub_dict.get("accuracy", 0) for sub_dict in all_metric])
    mcc_count = sum([sub_dict.get("mcc", 0) for sub_dict in all_metric])
    return f' acc:{(acc_count/args.show_loss_step) * 100}, mcc:{(mcc_count/args.show_loss_step) * 100}'