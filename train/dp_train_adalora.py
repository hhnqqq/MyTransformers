import torch
from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import DeepSpeedEngine

from common.utils import to_device, reduce_tensor
from dataset_classes import RepeatingLoader
from contextlib import contextmanager
import deepspeed

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


        # reference: https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/model.py#L238
        # orth_reg_weight = self.peft_config[self.trainable_adapter_name].orth_reg_weight
        orth_reg_weight = 0.5

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

def eval_step_deepspeed(model: DeepSpeedEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with record_function("eval_path"):
        batch = next(data_loader)
        batch = to_device(batch, args.device)
        with torch.no_grad():
            loss, metric = model(**batch)
            # TODO: all reduce metrics

            orth_reg_weight = 0.5

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

        return loss.item(), metric
    
def task_print_deepspeed(all_metric, args):
    # return the on-chip accuracy
    acc_count = sum([sub_dict.get("accuracy", 0) for sub_dict in all_metric])
    mcc_count = sum([sub_dict.get("mcc", 0) for sub_dict in all_metric])
    return f' acc:{(acc_count/args.show_loss_step) * 100}, mcc:{(mcc_count/args.show_loss_step) * 100}'


@contextmanager
def gather_params_ctx(param, modifier_rank: int = 0, fwd_module: torch.nn.Module = None):
    """Call DeepSpeed GatheredParameters context manager if DeepSpeed is enabled, otherwise do nothing."""

    with deepspeed.zero.GatheredParameters(param, modifier_rank=modifier_rank, fwd_module=fwd_module):
        yield
    return


def update_and_allocate(model, global_step, saved_gradients):
    """
    This method updates Adalora budget and mask.

    This should be called in every training step after `loss.backward()` and before `zero_grad()`.

    `tinit`, `tfinal` and `deltaT` are handled with in the method.

    Args:
        global_step (`int`): The current training step, it is used to calculate adalora budget.

    Example:

    ```python
    >>> loss = model(**input).loss
    >>> loss.backward()
    >>> optimizer.step()
    >>> model.base_model.update_and_allocate(i_step)
    >>> optimizer.zero_grad()
    ```
    """
    lora_config = model.rankallocator.peft_config
    # Update the importance score and allocate the budget
    if global_step < lora_config.num_micro_update_steps - lora_config.tfinal:
        _, rank_pattern = model.rankallocator.update_and_allocate(model, global_step, saved_gradients)
        if rank_pattern:
            lora_config.rank_pattern = rank_pattern
    # Finalize the budget allocation
    elif global_step == lora_config.num_micro_update_steps - lora_config.tfinal:
        _, rank_pattern = model.rankallocator.update_and_allocate(model, global_step, saved_gradients, force_mask=True)
        # for some reason, this freezes the trainable parameters and nothing gets updates
        # self.resize_modules_by_rank_pattern(rank_pattern, self.trainable_adapter_name)
        lora_config.rank_pattern = rank_pattern
        model.rankallocator.reset_ipt()
    # Currently using inefficient way to mask the unimportant weights using the rank pattern
    #  due to problem mentioned above
    elif global_step > lora_config.num_micro_update_steps - lora_config.tfinal:
        model.rankallocator.mask_using_rank_pattern(model, lora_config.rank_pattern)
    # Pass the function and do forward propagation
    else:
        return None