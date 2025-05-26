"""Implementation of LoRA-GA-Pro
Code reference: https://github.com/Outsider565/LoRA-GA/blob/main/peft/src/peft/tuners/lora/layer.py
"""
import os
import random
import json
from typing import Callable

from common.utils import print_rank_0, reduce_tensor, Timer, ensure_directory_exists
from common.utils import to_device
from common.lora_modules.lora import *

class LinearWithLoRAGAPro(LinearWithLoRA):
    def __init__(self, 
                 lora_config: LoRAConfig,
                 rank_stablize: bool=False,
                 dynamic_scaling: bool=False):
        super().__init__(lora_config)
        self.rank_stablize = rank_stablize
        self.dynamic_scaling = dynamic_scaling
        self.scaling_alpha = lora_config.lora_scaler

    def prepare_init(self, allocated_rank: int = 0):
        # Get lora scaler first
        self._get_scalling(self.lora_rank, allocated_rank)

        assert allocated_rank !=0, f"Allocated rank should not be 0, but got {allocated_rank}."
        if allocated_rank * 2 > torch.min(torch.tensor(self.weight.shape)):
            allocated_rank = torch.min(torch.tensor(self.weight.shape)) // 2
            print_rank_0(f"LoRA rank {allocated_rank} is too large for the weight matrix of shape {self.weight.shape}. Setting rank to {allocated_rank}.", global_rank)
        self.lora_rank = allocated_rank
        
    def gradient_reinit(self, 
                        direction: str = 'ArB2r', 
                        scale: str = 'gd', 
                        stable_gamma: int = 16, 
                        scaling_factor: int = 16,
                        global_rank: int = 0,
                        is_first: bool = False,
                        reset_weight: bool = False,
                        allocated_rank: int = 0):
        """
        Reinitialize the LoRA weights based on the gradient of the original weight matrix.

        This method implements the core functionality of LoRA-GA (Gradient-based Adaptation).
        It performs SVD on the weight gradient and uses the resulting matrices to update
        the LoRA weights (A and B).

        Args:
            direction (str): Determines how to select A and B from SVD results.
                Options: 'ArBr', 'A2rBr', 'ArB2r'. Default is 'ArB2r'.
            scale (str): Scaling method for the new LoRA weights.
                Options: 'gd', 'unit', 'stable', 'weightS'. Default is 'stable'.
            stable_gamma (float): Gamma parameter for 'stable' scaling. Default is 16.

        The method performs the following steps:
        1. Compute SVD of the weight gradient
        2. Select A and B matrices based on the 'direction' parameter
        3. Apply scaling to A and B based on the 'scale' parameter
        4. Update the LoRA weights (weight_a and weight_b)

        Note: This method assumes that the LinearWithLora layer has gradient. Please call this
        method in the first step of training(before the model.step() call, or the gradient will be cleared.)
        """
        self.prepare_init(allocated_rank=allocated_rank)

        if not hasattr(self.weight, 'grad_stored'):
            return

        # Perform SVD on the weight gradient
        # Weight stored gradient shape [out_feature, in_feature]
        U, S, V = torch.svd_lowrank(self.weight.grad_stored.float().cuda(), 
                                    q = 4 * self.lora_rank if 4 * self.lora_rank < torch.min(torch.tensor(self.weight.shape)) else torch.min(torch.tensor(self.weight.shape)),
                                    niter=4)
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
            _, S, _ = torch.svd_lowrank(self.weight.float(), q=4 * self.lora_rank, niter=4)
            S /= scaling_factor
            avg_s = torch.sqrt(S[:self.lora_rank]).mean().to(A.device)
            A *= avg_s
            B *= avg_s
        elif scale != "unit":
            raise ValueError(f"Unknown scale: {scale}")
        del self.weight.grad_stored

        # Update the LoRA weights
        self.weight_a.data = A.contiguous().cuda()
        self.weight_b.data = B.contiguous().cuda()
        
        if reset_weight:
            self.reset_weight()

        if is_first:
            print_rank_0(f'--->LoRA-GA re-init example: weight_B->{self.weight_b}', global_rank)

    def reset_weight(self):
        weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

    def _get_scalling(self, lora_rank, allocated_rank) -> torch.Tensor:
        if self.dynamic_scaling:
            lora_rank = allocated_rank
        else:
            lora_rank = self.lora_rank
        if self.rank_stablize:
            lora_rank = lora_rank ** 0.5
        self.lora_scaler = torch.tensor(self.scaling_alpha / lora_rank).to(self._get_lora_dtype())


def get_record_gradient_hook(param):
    def record_gradient_hook(grad):
        torch.cuda.synchronize()
        if not hasattr(param, 'grad_stored'):
            param.grad_stored = grad.detach().cpu().clone()
        else:
            param.grad_stored += grad.detach().cpu()
        return grad

    return record_gradient_hook

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

def lora_ga_pro_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
    task_name: str = '',
    forward_backward_func: Callable = None
):
    r"""
    Compute the full-rank gradient of the model on the given dataset and reinitialize the LoRA weights.

    LoRA-GA Algorithm:
    1. Perform forward pass to get predictions
    2. Calculate loss
    3. Set learning rate η = α * sqrt(r)
    4. For each layer l from L to 1:
        a. Compute gradient ∇Wl ℓ
        b. Get layer dimensions dout, din
        c. Perform SVD: U, S, V = svd(∇Wl ℓ)
        d. Initialize Al using first r columns of V
        e. Initialize Bl using columns r+1 to 2r of U
        f. Update weights: Wl = Wl - η * Bl * Al
        g. Clear gradient for this layer

    Input:
    - Model f(·) with L layers and parameters W
    - Sampled batch B = {x, y}
    - LoRA rank r
    - LoRA scaling factor α
    - Loss function L
    - Scale factor γ

    Output:
    - Initialized parameters W, η, A, B

    Note: This implementation follows the LoRA-GA paper's algorithm closely.
    """
    print_rank_0("--->Estimating gradient for lora ga.", rank=args.global_rank)
    torch.cuda.empty_cache()
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for LoRA-GA layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(module.weight))
            for name, module in model.named_modules()
            if isinstance(module, LinearWithLoRAGAPro)
        ]

        for module in model.modules():
            if isinstance(module, LinearWithLoRAGAPro):
                module.weight.requires_grad = True
            elif isinstance(module, torch.nn.Linear):
                module.weight.requires_grad = False

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            if args.huggingface:
                loss = model(input_ids=batch['input_ids'],
                        labels=batch['labels'],
                        attention_mask=batch['attention_mask']).loss
            else:
                output = model(**batch)
                loss = output[0]
            loss.backward()
            print_rank_0(f'--->LoRA-GA gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)} ', args.global_rank)
            for p in model.parameters():
                p.grad = None

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        torch.cuda.empty_cache()
        torch.distributed.barrier()

        print_rank_0('--->All reduce LoRA-GA stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):

                p.grad_stored /= iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to('cpu')

        allocated_rank = {}
        print_rank_0('--->Computing erank for LoRA-GA-Pro', args.global_rank)
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRAGAPro):
                if not hasattr(module.weight, 'grad_stored'):
                        print_rank_0(f'--->Module: {name} does not have stored gradients', args.global_rank)
                        continue
                rank = compute_effective_rank(module.weight.grad_stored, dtype=torch.float32)
                # The min rank is 4, and the max rank is 512.
                # The rank is divided by 2 because the LoRA-GA-Pro uses two matrices A and B.
                rank = max(4, min(int(rank / 2), 512))
                allocated_rank[name] = rank
                print_rank_0(f'--->LoRA rank {rank} is already allocated for module: {name}.', args.global_rank)

        save_folder = os.path.join(args.output_path, args.experiment_name)
        if task_name:
            save_folder = os.path.join(save_folder, task_name)
        
        ensure_directory_exists(save_folder, args.global_rank)
        if args.global_rank == 0:
            with open(os.path.join(save_folder, 'rank.json'), 'w') as f:
                json.dump(allocated_rank, f)

        is_first = True
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRAGAPro):
                print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
                module.gradient_reinit(scale=args.lora_ga_scale_method, 
                                       global_rank=args.global_rank, 
                                       is_first=is_first,
                                       reset_weight=args.lora_ga_reset_weight,
                                       allocated_rank=allocated_rank[name])
                is_first = False

    print_rank_0(f'--->Total time consumed for LoRA-GA initialization: {timer.time_cost}', args.global_rank)