# @author: haonan he
# @date: 2024-08-21
""" Implements RELORA"""

from common.lora_modules.lora import *
from common.utils.utils import print_rank_0
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

@torch.no_grad()
def random_pruning_(tensor, prune_ratio):
    """
    Performs random pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    random_pruning_mask = torch.rand_like(tensor) > prune_ratio
    tensor.mul_(random_pruning_mask)


@torch.no_grad()
def magnitude_pruning_(tensor, prune_ratio):
    """
    Performs magnitude pruning dimensionality reduction **inplace**.
    Only reduces the inner dimensionality, does not affect the shape of the tensor
    """
    tensor_magnitude = torch.abs(tensor)
    threshold = torch.quantile(tensor_magnitude.flatten().to(dtype=torch.float32), prune_ratio).to(dtype=tensor.dtype)

    mask = tensor_magnitude > threshold
    tensor.mul_(mask.to(dtype=tensor.dtype))

def optimizer_reset(
    optimizer,
    lr_scheduler,
    relora_auto_warmup_steps,
    relora_auto_warmup_ratio,
    *,
    reset_params: list[torch.nn.Parameter],
    reset_optimizer_on_relora: bool,
    optimizer_random_pruning: float,
    optimizer_magnitude_pruning: float,
    args,
    optimizer_state_keys: list[str] = ["exp_avg", "exp_avg_sq"]
):
    """
        optimizer_state_keys: e.g., ["exp_avg", "exp_avg_sq"]
    """
    n_reset_types = (
        int(bool(reset_optimizer_on_relora))
        + int(bool(optimizer_random_pruning))
        + int(bool(optimizer_magnitude_pruning))
    )
    if n_reset_types != 1:
        print_rank_0(f"Got {reset_optimizer_on_relora=}, {optimizer_random_pruning=}, {optimizer_magnitude_pruning=}",
                     args.global_rank)
        raise ValueError(f"Exactly one of reset_optimizer_on_relora, "
                         f"optimizer_random_pruning, optimizer_magnitude_pruning must be True")

    # pruning_fn has to be inplace to work with DeepSpeedZeroOptimizer
    if reset_optimizer_on_relora:
        print_rank_0("--->Resetting optimizer states to zeros", args.global_rank)
        # looks like zeroing out breaks dictionary in the optimizer
        # see full error below
        pruning_fn = partial(random_pruning_, prune_ratio=0.999)
    elif optimizer_random_pruning:
        print_rank_0(f"--->Performing random pruning of optimizer states. Pruning {optimizer_random_pruning} percent",
                     args.global_rank)
        pruning_fn = partial(random_pruning_, prune_ratio=optimizer_random_pruning)
    elif optimizer_magnitude_pruning:
        print_rank_0(f"--->Performing magnitude pruning of optimizer states. Pruning {optimizer_magnitude_pruning} percent",
                     args.global_rank)
        pruning_fn = partial(magnitude_pruning_, prune_ratio=optimizer_magnitude_pruning)
    else:
        raise ValueError("Unknown pruning type")
    n_zeros = 0
    n_total = 0

    optimizer_state = optimizer.state
    if isinstance(optimizer, DeepSpeedZeroOptimizer):
        optimizer_state = optimizer.optimizer.state

    for p in reset_params:
        param_state = optimizer_state[p]
        if len(param_state) == 0: # no state for this param, happens for ZeRo optimizer
            continue
        for key in optimizer_state_keys:
            pruning_fn(param_state[key])  # pruning fn has to be inplace to keep the same keys in the dict
            n_total += param_state[key].numel()
            n_zeros += torch.sum(param_state[key] == 0).item()

    _zeroed = n_zeros / (1e-7 + n_total) * 100
    print_rank_0(f"--->Percent of optimizer states zeroed: {_zeroed:.2f}", args.global_rank)
    # start a new warmup
    lr_scheduler.num_iters = 1
    lr_scheduler.auto_warmup_steps = relora_auto_warmup_steps
    lr_scheduler.auto_warmup_ratio = relora_auto_warmup_ratio
    