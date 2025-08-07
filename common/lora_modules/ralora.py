# author: jingqi ye
# Gradient Intrinsic Dimensionality Alignment：Narrowing The Gap Between Low-Rank Adaptation and Full Fine-Tuning
import os
import math
import json
import time

import pickle
import torch.distributed as dist
from torch import svd_lowrank as fast_svd

from typing import Callable
from collections import OrderedDict

from common.lora_modules.melora import *
from common.utils.utils import Timer, to_device, print_rank_0, ensure_directory_exists
from common.lora_modules.grad_utils import get_record_gradient_hook, broadcast_object

class LinearWithRaLoRA(LinearWithMELoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        ralora_dynamic_scaling: bool = False,
        ralora_rank_stablize: bool = False,
        forward_method: str = 'for'
    ):  
        self.dynamic_scaling = ralora_dynamic_scaling
        self.scaling_alpha = lora_config.lora_scaler
        self.rank_stablize = ralora_rank_stablize
        LinearWithQLoRA.__init__(self, lora_config)
        self.quant_after_init = self.quant
        self.quant = False

        self.lora_config = lora_config
        self.forward_method = forward_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return LinearWithQLoRA.forward(self, x)
    
    def _check_exact_division(self):
        if self.in_features % self.n_split != 0:
            raise ValueError(f"in_features ({self.in_features}) must be divisible by melora_n_split ({self.n_split})")
        if self.out_features % self.n_split != 0:
            raise ValueError(f"out_features ({self.out_features}) must be divisible by melora_n_split ({self.n_split})")
        
    def _prepare_ralora_attrs(self, lora_rank, in_features, out_features):
        self.lora_rank = lora_rank * self.n_split    
        self.in_features = in_features
        self.out_features = out_features

        self._check_exact_division()
        self.mini_lora_rank = int(lora_rank)
        self.mini_in_features = int(self.in_features / self.n_split)
        self.mini_out_features = int(self.out_features / self.n_split)

    def init_lora_weights(self):
        pass
    
    def _get_scaling(self, avg_rank, real_rank):
        if self.dynamic_scaling:
            self.scale_rank = real_rank
        else:
            self.scale_rank = avg_rank

        if self.rank_stablize:
            self.scale_rank = self.scale_rank**0.5
        self.lora_scaler = self.scaling_alpha / self.scale_rank
    
    def dynamic_init(self, avg_rank, rank, n_split=None):
        if n_split >= 1:
            self.n_split = n_split
        else:
            raise ValueError(f"The value of n_split: {n_split} must be greater than or equal to 1")
        
        if self.forward_method == "for":
            init_func = self.init_lora_weights_for
            self._lora_forward = self._lora_forward_for
        elif self.forward_method == "einsum":
            init_func = self.init_lora_weights_einsum
            self._lora_forward = self._lora_forward_einsum
        elif self.forward_method == "concat":
            init_func = self.init_lora_weights_for
            self._lora_forward = self._lora_forward_concat

        if n_split == 1:
            init_func = partial(LinearWithLoRA.init_lora_weights, self)
            self._lora_forward = partial(LinearWithLoRA._lora_forward, self)

        self._prepare_ralora_attrs(rank, 
                                   self.lora_config.in_features, 
                                   self.lora_config.out_features)
        
        if rank != 0:
            self._get_scaling(avg_rank, rank)
            with torch.no_grad():
                init_func()
            if hasattr(self.weight, "grad_stored"):
                del self.weight.grad_stored
            if hasattr(self.weight, "iters"):
                del self.weight.iters
            if self.quant_after_init:
                self.quant = True

def compute_importance(param, grad_stored):
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)
    importance = torch.mean(torch.abs(param * grad_stored)).item()
    return importance
    
def get_normalized_importances(importances_tensor):
    normalized_importances = importances_tensor / importances_tensor.sum()
    return normalized_importances

def get_allocated_rank(model, args):
    named_ranks = {}
    named_importances = OrderedDict()
    total_budget, smooth_total_budget, actual_trainable = 0, 0, 0
    named_features, named_smooth_features = {}, {}

    feature_adjust_func: Callable = {
        'sqrt': math.sqrt,
        'log1p': math.log1p,
        None: lambda x: x
    }.get(args.ralora_features_func, lambda x: x)

    if args.global_rank == 0:
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LinearWithRaLoRA):
                    if not hasattr(module.weight, 'grad_stored'):
                        print_rank_0(f'--->Module: {name} do not have stored gradients', args.global_rank)
                        continue
                    features = module.in_features + module.out_features
                    # Move gradient to GPU one at a time
                    grad_stored = module.weight.grad_stored.to(args.device)
                    importance = compute_importance(
                        module.weight.data,
                        grad_stored
                    )
                    named_importances[name] = importance
                    adjusted_features = feature_adjust_func(features)
                    named_smooth_features[name] = adjusted_features
                    named_features[name] = features
                    smooth_total_budget += adjusted_features * args.lora_rank
                    total_budget += features * args.lora_rank
                    # Clear GPU gradient after use
                    grad_stored = None
                    torch.cuda.empty_cache()

            if not named_importances:
                raise ValueError("No gradients were stored. Check if backward pass was performed correctly.")

            importances_tensor = torch.tensor(list(named_importances.values()))
            normalized_importances = get_normalized_importances(importances_tensor)

            for name, normalized_importance in zip(named_importances.keys(), normalized_importances):
                smooth_trainable = round(smooth_total_budget * normalized_importance.item())
                rank = smooth_trainable // named_smooth_features[name]
                if args.ralora_max_rank and args.ralora_min_rank:
                    named_ranks[name] = min(max(round(rank), args.ralora_min_rank), args.ralora_max_rank)
                else:
                    named_ranks[name] = rank
                actual_trainable += named_ranks[name] * named_features[name]

    else:
        total_budget, actual_trainable, named_importances = 0, 0, OrderedDict()
        named_ranks = {}
    # Broadcast named_ranks and has_converged to all ranks
        
    if args.world_size > 1:
        dist.barrier()
        
        if args.global_rank == 0:
            broadcast_data = {
                'named_ranks': named_ranks
            }
            serialized = pickle.dumps(broadcast_data)
            data = torch.ByteTensor(list(serialized)).to(args.device)
            length = torch.tensor([len(serialized)], dtype=torch.long, device=args.device)
        else:
            length = torch.tensor([0], dtype=torch.long, device=args.device)
            data = torch.empty(1024*1024, dtype=torch.uint8, device=args.device)  # Allocate sufficient space
            
        dist.broadcast(length, src=0)
        
        if args.global_rank != 0:
            data = torch.empty(length.item(), dtype=torch.uint8, device=args.device)
        dist.broadcast(data, src=0)
        
        if args.global_rank != 0:
            serialized = bytes(data.cpu().tolist())
            broadcast_data = pickle.loads(serialized)
            named_ranks = broadcast_data['named_ranks']
    return total_budget, actual_trainable, named_ranks, named_importances

def compute_effective_rank(gradient_matrix, dtype=torch.float32, eps=1e-10):
    """
    Compute the effective rank of a gradient matrix using the method from
    "THE EFFECTIVE RANK: A MEASURE OF EFFECTIVE DIMENSIONALITY" (Roy & Vetterli, 2007).

    Args:
        gradient_matrix (torch.Tensor): Input gradient matrix (2D tensor, shape: [m, n]).
        eps (float): Small value to avoid numerical instability in log computation (default: 1e-10).

    Returns:
        float: Effective rank of the gradient matrix.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)
    # Ensure the input is a 2D tensor
    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be a 2D tensor")

    # Perform Singular Value Decomposition (SVD)
    try:
        U, S, Vh = torch.linalg.svd(gradient_matrix)
    except RuntimeError as e:
        print(f"SVD computation failed: {e}")
        return 1.0  # Return minimal effective rank in case of failure

    # If no valid singular values, return minimal effective rank
    if S.numel() == 0:
        print('Some thing wrong, because the number of S=0')
        return 1.0

    # Compute L1 norm of singular values
    l1_norm = torch.sum(S)

    # Compute normalized singular values (p_k = sigma_k / ||sigma||_1)
    p = S / l1_norm

    # Compute Shannon entropy: H = -sum(p_k * log(p_k))
    # Add eps to avoid log(0)
    entropy = -torch.sum(p * torch.log(p + eps))

    # Compute effective rank: erank = exp(H)
    effective_rank = torch.exp(entropy).item()
    
    del U, S, Vh, gradient_matrix
    # Ensure effective rank is at least 1
    return max(1.0, effective_rank)


def count_singular_values_above_threshold(gradient_matrix: torch.Tensor, threshold: float = 1e-2, dtype=torch.float32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)

    if gradient_matrix.dim() != 2:
        raise ValueError("Input of gradient_matrix must be 2D tensor")

    try:
        _, S, _ = fast_svd(gradient_matrix, q=min(gradient_matrix.shape))
    except RuntimeError as e:
        return 1.0

    count = torch.sum(S > threshold).item()

    del S, gradient_matrix

    return max(1.0, count)

def count_singular_values_by_variance_threshold(gradient_matrix: torch.Tensor, cumulative_variance_threshold: float = 0.99, dtype=torch.float32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gradient_matrix = gradient_matrix.to(dtype=dtype, device=device)

    if gradient_matrix.dim() != 2:
        raise ValueError("Input gradient_matrix must be 2D tensor")

    if not (0.0 <= cumulative_variance_threshold <= 1.0):
        raise ValueError("cumulative_variance_threshold must in [0.0, 1.0]")

    try:
        _, S, _ = fast_svd(gradient_matrix, q=min(gradient_matrix.shape))
    except RuntimeError as e:
        return 0

    if S.numel() == 0:
        return 1.0

    singular_values_squared = S.pow(2)
    
    total_variance = torch.sum(singular_values_squared)

    if total_variance < 1e-10:
        return 1.0

    variance_ratios = singular_values_squared / total_variance

    cumulative_variance_ratios = torch.cumsum(variance_ratios, dim=0)

    indices_above_threshold = (cumulative_variance_ratios >= cumulative_variance_threshold).nonzero(as_tuple=True)[0]

    if indices_above_threshold.numel() == 0:
        return S.numel()
    else:
        count = indices_above_threshold.min().item() + 1
        
    del S, singular_values_squared, total_variance, variance_ratios, cumulative_variance_ratios, gradient_matrix # 释放不再需要的内存

    return max(1.0, count)

def compute_n_split_allocations(model, named_ranks, args):
    """
    Compute the optimal number of mini LoRA modules for each layer
    based on gradient importance.
    
    Returns a dictionary mapping module names to the number of 
    mini LoRA modules to use for that layer.
    """

    named_n_splits = {}
    named_eranks = {}
    print_rank_0(f'--->Allocate n using the erank method.', args.global_rank)
    print_rank_0(f'--->The max power is {args.ralora_erank_max_power}.', args.global_rank)
    min_power = 0
    max_power = args.ralora_erank_max_power
    if args.global_rank == 0:
        start_time = time.time()
        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA):
                if not hasattr(module.weight, 'grad_stored'):
                        print_rank_0(f'--->Module: {name} does not have stored gradients', args.global_rank)
                        continue
                if args.ralora_svd_threshold > 0:
                    # Count the number of singular values above the threshold
                    print_rank_0(f'--->Module {name} is calculating erank using Threshold svd', args.global_rank)
                    erank = count_singular_values_above_threshold(module.weight.grad_stored, 
                                                                    threshold=args.ralora_svd_threshold, 
                                                                    dtype=torch.float32)
                elif args.ralora_cumulative_variance_threshold > 0:
                        # Count the number of singular values that contribute to the cumulative variance
                        print_rank_0(f'--->Module {name} is calculating erank using cumulative variance svd', args.global_rank)
                        erank = count_singular_values_by_variance_threshold(module.weight.grad_stored, 
                                                                            cumulative_variance_threshold=args.ralora_cumulative_variance_threshold, 
                                                                            dtype=torch.float32)
                else:
                    erank = compute_effective_rank(module.weight.grad_stored)
                named_eranks[name] = erank
        end_time = time.time()
        print_rank_0(f'--->Time consumption for calculating svd: {end_time-start_time:.6f}s', args.global_rank)
        if not named_eranks:
                print_rank_0(f'--->No gradient erank calculated for dynamic n allocation', args.global_rank)
    
        if args.ralora_allocate_by_erank:
            result = named_eranks
        else:
            # Allocating n according erank and lora rank
            assert args.ralora_erank_max_power is not None, f'The eran_max_power must be setted.'
            for name, erank in named_eranks.items():
                n_splits_power = min(max_power, max(min_power, math.floor(math.log2(erank) - math.log2(named_ranks[name]))))
                named_n_splits[name] = 2 ** n_splits_power
                print_rank_0(f'--->Module {name}: grad_erank={math.ceil(erank)},  n_split={named_n_splits[name]}', args.global_rank)
            result = named_n_splits
    if args.world_size > 1:
        dist.barrier()
        
        if args.global_rank == 0:
            broadcast_data = {
                'result': result
            }
            serialized = pickle.dumps(broadcast_data)
            data = torch.ByteTensor(list(serialized)).to(args.device)
            length = torch.tensor([len(serialized)], dtype=torch.long, device=args.device)
        else:
            length = torch.tensor([0], dtype=torch.long, device=args.device)
            data = torch.empty(1024*1024, dtype=torch.uint8, device=args.device)  # Allocate sufficient space
            
        dist.broadcast(length, src=0)
        
        if args.global_rank != 0:
            data = torch.empty(length.item(), dtype=torch.uint8, device=args.device)
        dist.broadcast(data, src=0)
        
        if args.global_rank != 0:
            serialized = bytes(data.cpu().tolist())
            broadcast_data = pickle.loads(serialized)
            result = broadcast_data['result']
    return result

def ralora_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
    task_name: str = '',
    forward_backward_func: Callable = None
):  
    n_samples = args.batch_size_per_gpu * args.world_size * iters
    print_rank_0(f"--->Estimating gradient for RaLoRA. Number of samples: {n_samples} to estimate gradients.", rank=args.global_rank)
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for RaLoRA layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model, args.world_size, args.global_rank))
            for module in model.modules()
            if isinstance(module, LinearWithRaLoRA)
        ]

        for module in model.modules():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, LinearWithRaLoRA):
                module.weight.requires_grad = True

        for idx, batch in enumerate(dataloader):
            timer.average_time("start")
            batch = to_device(batch, args.device)
            if forward_backward_func:
                loss = forward_backward_func(model, batch)
            else:
                loss = model(**batch)[0]
            loss.backward()
            timer.average_time("end")

            print_rank_0(f'--->RaLoRA gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)}, time_cost: {timer.loop_time:.2f}s', args.global_rank)

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        for p in model.parameters():
            p.grad = None
            
        if args.world_size > 1:
            torch.distributed.barrier()

        print_rank_0('--->All reduce RaLoRA stored gradients if needed.', args.global_rank)
        if args.global_rank == 0:
            for p in model.parameters():
                if hasattr(p, 'grad_stored'):
                    p.grad_stored = p.grad_stored / p.iters

        named_n_splits, named_importances = {}, {}
        if args.ralora_allocate_by_erank:
            named_ranks = compute_n_split_allocations(model, {}, args)
        else:
            total_budget, actual_trainable, named_ranks, named_importances = get_allocated_rank(model, args)
            
            # Compute and allocate optimal number of mini LoRA modules
            print_rank_0('--->Computing dynamic n allocation for Mini-LoRA blocks', args.global_rank)
            named_n_splits = compute_n_split_allocations(model, named_ranks, args)

            print_rank_0(f'--->RaLoRA total budget: {total_budget}, actual trainable: {actual_trainable}', args.global_rank)

        save_floder = os.path.join(args.output_path, args.experiment_name)
        if task_name:
            save_floder = os.path.join(save_floder, task_name)

        named_ranks = {k:math.ceil(v) for k, v in named_ranks.items()}
        ensure_directory_exists(save_floder, args.global_rank)
        if args.global_rank == 0:
            with open(os.path.join(save_floder, 'rank.json'), 'w') as f:
                json.dump(named_ranks, f)
            with open(os.path.join(save_floder, 'importance.json'), 'w') as f:
                json.dump(named_importances, f)
            if named_n_splits:
                with open(os.path.join(save_floder, 'n_splits.json'), 'w') as f:
                    json.dump({name: int(n) for name, n in named_n_splits.items()}, f)

        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA) and name in named_ranks.keys():
                n_split = named_n_splits.get(name, 1)
                print_rank_0(f'--->Module {name} is initiating lora weight, rank: {named_ranks[name]}, n_split: {n_split}', args.global_rank)
                module.dynamic_init(args.lora_rank, named_ranks[name], n_split=n_split)
        torch.cuda.empty_cache()

    print_rank_0(f'--->Total time consumed for RaLoRA initialization: {timer.time_cost}', args.global_rank)