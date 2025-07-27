# @author: haonan he
"""
Official implementation for GoRA: Gradient-driven Adaptive Low Rank Adaptation [arxiv preprint]
Paper link: https://arxiv.org/abs/2502.12171
Code reference: this is the offical implementaion

GoRA simultaneously adapts both the rank and initialization strategy within a unified framework. 
GoRA leverages gradient information during training to dynamically assign optimal ranks and 
initialize low-rank adapter weights in an adaptive manner. 
"""
import os
import math
import json
import random
import numpy as np
from typing import Callable
from collections import OrderedDict
from torch import Tensor, svd_lowrank as fast_svd
from torch.linalg import svd as standard_svd

from common.lora_modules.lora import *
from common.utils.utils import Timer, reduce_tensor, to_device, print_rank_0, ensure_directory_exists

def z_score_normalize(tensor):
    mean = tensor.mean()
    std = tensor.std()
    normalized_tensor = (tensor - mean) / std
    return normalized_tensor

def get_est_nuc_norm(tensor, rank):
    _, Sr, _ = fast_svd(tensor, rank, niter=8)
    return torch.sum(torch.log1p(Sr))

def solve_L1(A, G, max_iter=1000, rho=1.0, tol=1e-5):
    A = A.cpu().numpy()
    G = G.cpu().numpy()
    n, m = G.shape
    r, _ = A.shape
    
    B = np.zeros((n, r))
    Z = np.zeros((n, m))
    Y = np.zeros((n, m))
    
    AAT = A @ A.T
    GAT = G @ A.T
    L = np.linalg.cholesky(AAT + rho * np.eye(r))
    
    for k in range(max_iter):
        rhs = GAT + rho * (Z - Y) @ A.T
        B = np.linalg.solve(L.T, np.linalg.solve(L, rhs.T)).T
        
        R = G - B @ A + Y
        Z = np.sign(R) * np.maximum(np.abs(R) - 1/rho, 0)
        
        Y += G - B @ A - Z
        
        primal_residual = np.linalg.norm(G - B @ A - Z, 'fro')
        if primal_residual < tol:
            break
    
    return torch.from_numpy(B).cuda()

class LinearWithGoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        fast_svd_n_iters: Optional[int] = 16,
        gora_init_method: str = 'compress',
        gora_rank_stablize: bool = False,
        gora_dynamic_scaling: bool = False
    ):
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2
        self.init_method = gora_init_method
        self.dynamic_scaling = gora_dynamic_scaling
        self.rank_stablize = gora_rank_stablize
        self.scaling_alpha = lora_config.lora_scaler
        super().__init__(lora_config,)

    def init_lora_weights(self):
        self.lora_rank = 0

    def _get_scaling(self, avg_rank, real_rank):
        if self.dynamic_scaling:
            self.scale_rank = real_rank
        else:
            self.scale_rank = avg_rank

        if self.rank_stablize:
            self.scale_rank = self.scale_rank**0.5
        self.lora_scaler = self.scaling_alpha / self.scale_rank

    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        if self.lora_rank == 0:
            return result
        else:
            return super()._lora_forward(x, result)
        
    def dynamic_init(self, avg_rank, rank, stable_gemma=16, scaling_by_lr=False, lr=1e-3):
        """
        During inference, this should be called before loading checkpoint, 
        and set the init method to vanilla
        """
        if rank != 0:
            self._get_scaling(avg_rank, rank)
            self.avg_lora_rank = avg_rank
            self.lora_rank = rank
            with torch.no_grad():
                if self.init_method == 'weight_svd':
                    self.weight_svd_init()
                elif self.init_method == 'grad_svd':
                    self.grad_svd_init()
                elif self.init_method == 'compress':
                    self.grad_compress_init(stable_gemma=stable_gemma,
                                       scaling_by_lr=scaling_by_lr,
                                       lr=lr)
                elif self.init_method == 'vanilla':
                    super().init_lora_weights()
            if hasattr(self.weight, "grad_stored"):
                del self.weight.grad_stored
            if hasattr(self.weight, "iters"):
                del self.weight.iters

    def grad_compress_init(self, 
                      lr: float, 
                      scaling_by_lr: bool = False, 
                      stable_gemma: int = None, 
                      reinit_weight: bool = False,
                      weight_init_a: bool = False,
                      grad_init_a: bool = False,
                      l1=False):
        if not hasattr(self.weight, 'grad_stored'):
            return
        
        # Convert weight_a to float32 on correct device
        origin_weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        weight_dtype = weight.dtype
        weight_device = weight.device
        grad_stored = self.weight.grad_stored.to(weight_dtype).to(weight_device)
        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=weight_dtype, device=weight_device))
        if weight_init_a:
            Ur = fast_svd(weight.data, self.lora_rank, niter=16)[-1]
            Uhr = Ur.t()
            self.weight_a.data = Uhr.to(weight_dtype)
        elif grad_init_a:
            Ur = fast_svd(grad_stored, self.lora_rank, niter=16)[-1]
            Uhr = Ur.t()
            self.weight_a.data = Uhr.to(weight_dtype) * self.out_features**0.25 / 4
        else:
            self._init_weight('weight_a')
        if l1:
            weight_b_data = solve_L1(self.weight_a, grad_stored)
        else:
            AT = self.weight_a.T
            AAT = torch.matmul(self.weight_a, AT)
            
            AAT_inv = torch.linalg.pinv(AAT + 1e-8 * torch.eye(self.lora_rank).to(weight_device)).to(weight_dtype)
            AAT_inv_AT = torch.matmul(AT, AAT_inv)

            # Compute weight_b using grad_stored (convert to float32 for computation)
            weight_b_data = torch.matmul(grad_stored, AAT_inv_AT)

        if scaling_by_lr:
            stable_gemma = (lr / math.sqrt(self.lora_rank/self.in_features)) * (self.scale_rank)
        weight_b_data *= (stable_gemma / self.scaling_alpha)

        self.weight_b = nn.Parameter(-weight_b_data.contiguous())

        # Final weight update with proper dtype conversion
        if reinit_weight:
            updated_weight = weight - self._compute_lora_weight()
            self.weight.data = updated_weight.to(origin_weight_dtype)

        
    def grad_svd_init(self, 
                     direction: str = 'ArB2r', 
                     scale: str = 'stable', 
                     stable_gamma: int = 16, 
                     scaling_factor: int = 16):
        if not hasattr(self.weight, 'grad_stored'):
            return
        
        # Perform SVD on the weight gradient
        # Weight stored gradient shape [out_feature, in_feature]
        U, S, V = torch.svd_lowrank(self.weight.grad_stored.float().cuda(), q=4 * self.lora_rank, niter=4)
        # U shape [out_feature, 4r] S shape [4r, 4r] V shape [in_feature, 4r]
        V = V.T

        # Determine A and B based on the direction parameter
        if direction == "ArBr":
            # B shape [out_feature, r/2]
            B = U[:, 0:2 * self.lora_rank:2]
            A = V[1:2 * self.lora_rank:2, :]
        elif direction == "A2rBr":
            # B shape [out_feature, r]
            B = U[:, :self.lora_rank]
            # A shape [r, in_feature]
            A = V[self.lora_rank:2 * self.lora_rank, :]
        elif direction == "ArB2r":
            B = U[:, self.lora_rank:2 * self.lora_rank]
            A = V[:self.lora_rank, :]
        elif direction == "random":
            random_list = random.sample(range(2 * self.lora_rank), 2 * self.lora_rank)
            indexes_A = random_list[0:self.lora_rank]
            indexes_B = random_list[self.lora_rank:2 * self.lora_rank]
            B = U[:, indexes_B]
            A = V[indexes_A, :]
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Apply scaling to A and B based on the scale parameter
        if scale == "gd":
            A /= scaling_factor
            B /= scaling_factor
        elif scale == "stable":
            m, n = self.weight.grad_stored.shape 
            A = A * m**0.25 / stable_gamma**0.5
            B = B * m**0.25 / stable_gamma**0.5
        elif scale == "weightS":
            _, S, _ = fast_svd(self.weight.float(), q=4 * self.lora_rank, niter=4)
            S /= scaling_factor
            avg_s = torch.sqrt(S[:self.lora_rank]).mean().to(A.device)
            A *= avg_s
            B *= avg_s
        elif scale != "unit":
            raise ValueError(f"Unknown scale: {scale}")

        # Update the LoRA weights
        self.weight_a = nn.Parameter(A.contiguous().cuda())
        self.weight_b = nn.Parameter(B.contiguous().cuda())
        
        weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

    def weight_svd_init(self):
        if self.lora_rank > 0:
            dtype = self._get_lora_dtype()
            weight_dtype = self.weight.dtype

            weight = self.weight.to(torch.float32)
            if self.fast_svd:
                # Run fast svd.
                Vr, Sr, Ur = fast_svd(weight.data, self.lora_rank, niter=self.n_iters)
                Uhr = Ur.t()
            else:
                # Full svd, which is very slow.
                V, S, Uh = standard_svd(self.weight.data, full_matrices=False)
                Vr, Sr, Uhr = V[:, :self.lora_rank], S[:self.lora_rank], Uh[:self.lora_rank]

            Sr.div_(self.lora_scaler) 
            sqrt_Sr = Sr.sqrt_()
            
            weight_a_data = torch.diag(sqrt_Sr) @ Uhr
            self.weight_a = nn.Parameter(weight_a_data.to(dtype), requires_grad=True)
            weight_b_data = Vr @ torch.diag(sqrt_Sr)
            self.weight_b = nn.Parameter(weight_b_data.to(dtype), requires_grad=True)

            self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

def get_record_gradient_hook(model):
    def record_gradient_hook(grad):
        torch.cuda.synchronize()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                if not hasattr(p, 'grad_stored'):
                    p.grad_stored = p.grad.detach().cpu()
                    p.iters = 1
                else:
                    p.grad_stored += p.grad.detach().cpu()
                    p.iters += 1
                p.grad = None
        return grad
    return record_gradient_hook

def compute_importance(param, grad_stored, features, scale_features, type, lora_rank, max_lora_rank):
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)
    if max_lora_rank:
        rank = max_lora_rank
    else:
        rank = 4 * lora_rank
    if type == 'union_frobenius_norm':
        importance = torch.linalg.matrix_norm(param * grad_stored).item()
    elif type == 'union_2ord_norm':
        importance = torch.mean(torch.linalg.norm(param * grad_stored, dim=1)).item()
    elif type == 'union_mean':
        importance = torch.mean(torch.abs(param * grad_stored)).item()
    elif type == 'union_nuc_norm':
        importance = torch.linalg.matrix_norm(param * grad_stored, ord='nuc').item()
    elif type == 'grad_nuc_norm':
        importance = torch.linalg.matrix_norm(grad_stored, ord='nuc').item()
    elif type == 'grad_est_nuc_norm':
        importance = get_est_nuc_norm(grad_stored, rank).item()
    elif type == 'union_est_nuc_norm':
        importance = get_est_nuc_norm(param * grad_stored, rank).item()
    elif type == 'grad_frobenius_norm':
        importance = torch.linalg.matrix_norm(grad_stored).item()
    elif type == 'grad_mean':
        importance = torch.mean(torch.abs(grad_stored)).item()
    elif type == 'union_mean_grad_nuc_norm':
        importance = (torch.mean(torch.abs(param * grad_stored)).item(),
                      get_est_nuc_norm(grad_stored, rank).item())
    elif type == 'union_mean_union_nuc_norm':
        param_grad = param * grad_stored
        importance = (torch.mean(torch.abs(param_grad)).item(),
                      get_est_nuc_norm(param_grad, rank).item())
    elif type == 'grad_mean_grad_nuc_norm':
        importance = (torch.mean(torch.abs(grad_stored)).item(),
                      get_est_nuc_norm(grad_stored, rank).item())

    if scale_features:
        if isinstance(importance, tuple):
            importance = tuple(math.sqrt(i) for i in importance)
        else:
            importance = math.sqrt(importance)
    return isinstance(importance, tuple), importance

def get_normalized_importances(args, importances_tensor):
    if args.gora_softmax_importance:
        normalized_importances = torch.softmax(
            (importances_tensor - importances_tensor.min()) /
            (importances_tensor.max() - importances_tensor.min()) / args.gora_temperature,
            dim=0
        )
    else:
        normalized_importances = importances_tensor / importances_tensor.sum()
    return normalized_importances

def get_allocated_rank(model, args, prev_importances=None):
    named_ranks = {}
    named_importances = OrderedDict()
    total_budget, smooth_total_budget, actual_trainable = 0, 0, 0
    named_features, named_smooth_features = {}, {}

    allocate_func: Callable = {
        'radical': math.ceil,
        'moderate': round,
        'conserved': math.floor
    }.get(args.gora_allocate_stretagy, round)

    feature_adjust_func: Callable = {
        'sqrt': math.sqrt,
        'log1p': math.log1p,
        None: lambda x: x
    }.get(args.gora_features_func, lambda x: x)

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LinearWithGoRA):
                if not hasattr(module.weight, 'grad_stored'):
                    print_rank_0(f'--->Module: {name} do not have stored gradients', args.global_rank)
                    continue
                features = module.in_features + module.out_features
                # Normalize gradients by iters and average across GPUs
                grad_stored = module.weight.grad_stored / module.weight.iters
                if args.world_size > 1:
                    grad_stored = reduce_tensor(grad_stored.to(args.device), args.world_size)
                is_tuple, importance = compute_importance(
                    module.weight.data,
                    grad_stored,
                    features,
                    args.gora_scale_importance,
                    args.gora_importance_type,
                    args.lora_rank,
                    args.gora_max_rank
                )
                named_importances[name] = importance
                adjusted_features = feature_adjust_func(features)
                named_smooth_features[name] = adjusted_features
                named_features[name] = features
                smooth_total_budget += adjusted_features * args.lora_rank
                total_budget += features * args.lora_rank

        if not named_importances:
            raise ValueError("No gradients were stored. Check if backward pass was performed correctly.")

        if is_tuple:
            first_importances_tensor = torch.tensor([i[0] for i in list(named_importances.values())])
            second_importances_tensor = torch.tensor([i[1] for i in list(named_importances.values())])
            first_normalized_importances = get_normalized_importances(args, first_importances_tensor)
            second_normalized_importances = get_normalized_importances(args, second_importances_tensor)
            normalized_importances = torch.tensor([0.5 * a + 0.5 * b for a, b in zip(first_normalized_importances,
                                                                                     second_normalized_importances)])
        else:
            importances_tensor = torch.tensor(list(named_importances.values()))
            normalized_importances = get_normalized_importances(args, importances_tensor)

        for name, normalized_importance in zip(named_importances.keys(), normalized_importances):
            smooth_trainable = allocate_func(smooth_total_budget * normalized_importance.item())
            rank = smooth_trainable // named_smooth_features[name]
            if args.gora_max_rank and args.gora_min_rank:
                named_ranks[name] = min(max(allocate_func(rank), args.gora_min_rank), args.gora_max_rank)
            else:
                named_ranks[name] = rank
            actual_trainable += named_ranks[name] * named_features[name]

        # Check for convergence
        has_converged = False
        if prev_importances is not None:
            all_converged = True
            for name in named_importances:
                if name not in prev_importances:
                    all_converged = False
                    break
                curr_imp = named_importances[name]
                prev_imp = prev_importances[name]
                if isinstance(curr_imp, tuple):
                    for c, p in zip(curr_imp, prev_imp):
                        if abs(c - p) / (p + 1e-8) > args.gora_convergence_threshold:
                            all_converged = False
                            break
                else:
                    if abs(curr_imp - prev_imp) / (prev_imp + 1e-8) > args.gora_convergence_threshold:
                        all_converged = False
                        break
            has_converged = all_converged

    return total_budget, actual_trainable, named_ranks, named_importances, has_converged

def gora_dynamic_init(model, args, named_ranks, batch):
    with torch.no_grad():
        target_modules = [
            (name, module) for name, module in model.named_modules()
            if isinstance(module, LinearWithGoRA) and name in named_ranks
        ]
        
        gora_stable_gemma = args.gora_stable_gemma if not args.gora_adaptive_lr_selection else 1
        gora_scale_by_lr = args.gora_scale_by_lr if not args.gora_adaptive_lr_selection else False
        with Timer() as init_timer:
            for name, module in target_modules:
                print_rank_0(f'--->Module {name} is initiating lora weight, rank is: {named_ranks[name]}', args.global_rank)
                module.dynamic_init(args.lora_rank, named_ranks[name], gora_stable_gemma, gora_scale_by_lr, lr=args.gora_lr)
        print_rank_0(f"--->Total time cost for gora dynamic init: {init_timer.time_cost}")

        if args.gora_scale_by_lr and args.gora_adaptive_lr_selection:
            best_loss = float("inf")
            base_loss = model(**batch)[0]
            if args.world_size > 1:
                base_loss = reduce_tensor(base_loss.to(args.device), args.world_size)
            best_lr = 0.0
            print_rank_0(f'--->[GoRA lr selection] base_loss:{base_loss}', args.global_rank)

            lr_candidates = []
            current_lr = 1
            while current_lr >= 1e-5:
                lr_candidates.append(current_lr)
                current_lr *= 0.8
            
            if lr_candidates[-1] < 1e-5:
                lr_candidates.append(1e-5)
            
            for lr in lr_candidates:
                for name, module in target_modules:
                    scale = (lr / math.sqrt(module.lora_rank / module.in_features)) * module.scale_rank
                    module.weight_b *= scale
                
                current_loss = model(**batch)[0]
                if args.world_size > 1:
                    current_loss = reduce_tensor(current_loss.to(args.device), args.world_size)
                    print_rank_0(f'--->[GoRA lr selection] lr: {lr}, current_loss: {current_loss}', args.global_rank)
                
                if current_loss < best_loss and current_loss < base_loss:
                    best_loss = current_loss
                    best_lr = lr
                
                for name, module in target_modules:
                    inv_scale = 1 / ((lr / math.sqrt(module.lora_rank / module.in_features)) * module.scale_rank)
                    module.weight_b *= inv_scale
            
            print_rank_0(f'--->The best adaptive lr for gora is {best_lr}', args.global_rank)
            for name, module in target_modules:
                best_scale = (best_lr / math.sqrt(module.lora_rank / module.in_features)) * module.scale_rank
                module.weight_b *= best_scale

def gora_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
    task_name: str = '',
    forward_backward_func: Callable = None
):
    print_rank_0("--->Estimating gradient for gora.", rank=args.global_rank)
    torch.cuda.empty_cache()
    with Timer() as timer:
        model.to(args.device)
        model.train()

        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model))
            for module in model.modules()
            if isinstance(module, LinearWithGoRA)
        ]

        for module in model.modules():
            if isinstance(module, LinearWithGoRA):
                module.weight.requires_grad = True
            elif isinstance(module, torch.nn.Linear):
                module.weight.requires_grad = False

        prev_importances = None
        for idx, batch in enumerate(dataloader):
            timer.average_time("start")
            if idx >= iters:
                break
            batch = to_device(batch, args.device)
            if forward_backward_func:
                loss = forward_backward_func(model, batch)
            else:
                loss = model(**batch)[0]
            loss.backward()

            if args.gora_adaptive_n_selection:
                if idx + 1 >= args.gora_min_steps:
                    total_budget, actual_trainable, named_ranks, named_importances, has_converged = get_allocated_rank(model, args, prev_importances)
                    prev_importances = named_importances
                    if has_converged:
                        print_rank_0(f'--->All layers have converged at step {idx+1}. Stopping gradient accumulation.', args.global_rank)
                        break
                elif (idx + 1) < iters:
                    total_budget, actual_trainable, named_ranks, named_importances, _ = get_allocated_rank(model, args, prev_importances)
                    prev_importances = named_importances
            timer.average_time("end")
            print_rank_0(f'--->GoRA gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)}, time_cost: {timer.loop_time:.2f}s', args.global_rank)

        for hook in hooks:
            hook.remove()

        for p in model.parameters():
            p.grad = None
            
        if args.world_size > 1:
            torch.distributed.barrier()

        print_rank_0('--->All reduce GoRA stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):
                p.grad_stored = p.grad_stored / p.iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to("cpu")

        total_budget, actual_trainable, named_ranks, named_importances, _ = get_allocated_rank(model, args)

        save_floder = os.path.join(args.output_path, args.experiment_name)
        if task_name:
            save_floder = os.path.join(save_floder, task_name)
            
        ensure_directory_exists(save_floder, args.global_rank)
        if args.global_rank == 0:
            with open(os.path.join(save_floder, 'rank.json'), 'w') as f:
                json.dump(named_ranks, f)
            with open(os.path.join(save_floder, 'importance.json'), 'w') as f:
                json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in named_importances.items()}, f)

        print_rank_0(f'--->GoRA total budget: {total_budget}, actual trainable: {actual_trainable}', args.global_rank)
        gora_dynamic_init(model, args, named_ranks, to_device(next(iter(dataloader)), args.device))
        torch.cuda.empty_cache()

    print_rank_0(f'--->Total time consumed for GoRA initialization: {timer.time_cost}, Peak memory used: {timer.peak_memory}MB', args.global_rank)