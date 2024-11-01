"""Implementation of lora-pro 
Code reference: https://github.com/mrflogs/LoRA-Pro/blob/main/peta/optim.py#L3
This implementation only contain adamw implementation
"""
import math
import torch

from typing import Tuple, Union, List

from torch.optim import Optimizer
from torch._utils import is_compiling
from scipy.linalg import solve_sylvester
from common.utils import print_rank_0


def find_lora_names(n):
    for substring in ['weight_a', 'weight_b']:
        if substring in n:
            return substring
    return ""

def _dispatch_sqrt(
    x: float,
):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return math.sqrt(x)

def _get_value(x):
    # item is significantly faster than a cpu tensor in eager mode
    if not torch.jit.is_scripting() and is_compiling():
        return x
    else:
        return x.item() if isinstance(x, torch.Tensor) else x

def _get_scalar_dtype():
    return (
        torch.float64 if torch.get_default_dtype() == torch.float64 else torch.float32
    )

def solve_sylvester(A, B, C, X=None):
    ''' From the answer here: 
        https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch
    '''
    if A.dtype is torch.bfloat16:
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)
    B = -B
    m = B.shape[-1];
    n = A.shape[-1];
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = U[...,:n,:n] @ Y[...,:n,:m] @ torch.linalg.inv(V)[...,:m,:m]
    return X.real if all(torch.isreal(x.flatten()[0]) 
                for x in [A, B, C]) else X

class LoRAProAdamW(Optimizer):
    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        lr: Union[float, torch.Tensor] = 1e-3,
        scaling_factor: float = 2.,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
        X_mode: str = "sylvester",
        lora_plus_scaler: int = 1
    ):
        
        """
        Example of named params:
        [{'params':named_param_group1, 'lr':lr1},
        {'params':named_param_group2, 'lr':lr2}]
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not X_mode in ["zero", "sylvester", "symmetry"]:
            raise ValueError(f"Invalid mode value: {X_mode}, mode should be in ['zero', 'sylvester', 'symmetry']")

        self.X_mode = X_mode
        self.step_ = 0
        self.lora_plus_scaler = lora_plus_scaler
        
        if not isinstance(named_params, list):
            named_params = [named_params]
        # Process named_params into param groups
        params = []
        name_shape_mapping = {}

        for named_params_group in named_params:
            param_group = {
                'params': [],
                'names': [],
                'lr': named_params_group.get('lr', lr),
                'scaling_factor': scaling_factor,
                'betas': betas,
                'eps': eps,
                'weight_decay': weight_decay,
                'amsgrad': amsgrad,
                'maximize': maximize,
                'differentiable': differentiable,
                'X_mode': X_mode
            }

            for name, param in named_params_group['params']:
                param_group['params'].append(param)
                param_group['names'].append(name)
                name_shape_mapping[name] = param.shape
                
            params.append(param_group)
        
        defaults = dict(
            lr=lr,
            scaling_factor=scaling_factor,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            differentiable=differentiable,
            X_mode=X_mode,
        )
        
        super().__init__(params, defaults)
               
    def is_same(self, name_list):
        return (name_list[0].split('.')[:-3] == name_list[1].split('.')[:-3])
    
    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        self._cuda_graph_capture_health_check()
        
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            self._update_group_params(group)

        return loss

    def _update_group_params(self, group):
        beta1, beta2 = group["betas"]
        scaling_factor = group["scaling_factor"]

        param_dict = {}
        for p, n in zip(group["params"], group["names"]):
            if p.grad is None:
                continue
            lora_weight_name = find_lora_names(n)
            if lora_weight_name:
                param_dict[lora_weight_name] = p
                if len(param_dict.keys()) == 2:
                    name = n[: n.find(lora_weight_name)] + 'lora'
                elif len(param_dict.keys()) == 1:
                    continue
            else:
                name = n
            
            state = self.state[name]
        
            if len(state) == 0:
                self._initialize_state(state, param_dict, p, group)

            if len(param_dict.keys()) == 2:
                self._update_lora_params(state, param_dict, group, scaling_factor)
            else:
                self._update_standard_params(state, p, group, beta1, beta2)

    def _initialize_state(self, state, param_dict, p, group):
        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
        if len(param_dict.keys()) == 2:
            self._initialize_lora_state(state, param_dict, p.device, p.dtype, group["amsgrad"])
        else:
            self._initialize_standard_state(state, p.shape, p.device, p.dtype, group["amsgrad"])

    def _initialize_lora_state(self, state, param_dict, device, dtype, amsgrad):
        state["exp_avg_A"] = torch.zeros(param_dict['weight_a'].shape).to(device).to(dtype)
        state["exp_avg_B"] = torch.zeros(param_dict['weight_b'].shape).to(device).to(dtype)
        state["exp_avg_sq_A"] = torch.zeros(param_dict['weight_a'].shape).to(device).to(dtype)
        state["exp_avg_sq_B"] = torch.zeros(param_dict['weight_b'].shape).to(device).to(dtype)

        if amsgrad:
            state["max_exp_avg_sq_A"] = torch.zeros(param_dict['weight_a'].shape).to(device).to(dtype)
            state["max_exp_avg_sq_B"] = torch.zeros(param_dict['weight_b'].shape).to(device).to(dtype)

    def _initialize_standard_state(self, state, shape, device, dtype, amsgrad):
        state["exp_avg"] = torch.zeros(shape).to(device).to(dtype)
        state["exp_avg_sq"] = torch.zeros(shape).to(device).to(dtype)
        
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros(shape).to(device).to(dtype)
    
    def _update_lora_params(self, state, param_dict, group, scaling_factor):
        A = param_dict['weight_a']
        B = param_dict['weight_b']
        grad_A_orin = A.grad
        grad_B_orin = B.grad
        
        delta = 1e-8
        AA_T = A @ A.T
        B_TB = B.T @ B
        AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(A.shape[0]).to(A.device)) 
        B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(A.shape[0]).to(A.device))

        X = self._compute_X(group, B, A, scaling_factor, grad_A_orin, grad_B_orin, B_TB_inv, AA_T, AA_T_inv)
        
        grad_A = (1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin + X @ A
        grad_B = (1 / scaling_factor ** 2) * ((torch.eye(B.shape[0]).to(B.device) - B @ B_TB_inv @ B.T) @ grad_B_orin @ AA_T_inv) - B @ X
        
        self._adamw_update(state, group, A, B, grad_A, grad_B)

    def _compute_X(self, group, B, A, scaling_factor, grad_A_orin, grad_B_orin, B_TB_inv, AA_T, AA_T_inv):
        if group['X_mode'] == "sylvester":
            return solve_sylvester(B.T @ B, A @ A.T, -(1 / scaling_factor ** 2) * B_TB_inv @ grad_A_orin @ A.T)
        elif group['X_mode'] == "symmetry":
            return -0.5 * (1 / scaling_factor ** 2) * B_TB_inv @ B.T @ grad_B_orin @ AA_T
        else:
            return torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0])).to(B.device)

    def _adamw_update(self, state, group, A, B, grad_A, grad_B):
        exp_avg_A = state["exp_avg_A"]
        exp_avg_sq_A = state["exp_avg_sq_A"]
        
        exp_avg_B = state["exp_avg_B"]
        exp_avg_sq_B = state["exp_avg_sq_B"]

        step_t = state["step"]

        step_t += 1

        exp_avg_A.lerp_(grad_A, 1 - group["betas"][0])
        exp_avg_B.lerp_(grad_B, 1 - group["betas"][0])
        exp_avg_sq_A.mul_(group["betas"][1]).addcmul_(grad_A, grad_A.conj(), value=1 - group["betas"][1])
        exp_avg_sq_B.mul_(group["betas"][1]).addcmul_(grad_B, grad_B.conj(), value=1 - group["betas"][1])

        step = _get_value(step_t)
        
        bias_correction1 = 1 - group["betas"][0]**step
        bias_correction2 = 1 - group["betas"][1]**step

        step_size = group['lr'] 
        step_size_b = self.lora_plus_scaler * step_size

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        
        if group['amsgrad']:
            torch.maximum(state["max_exp_avg_sq_A"], exp_avg_sq_A, out=state["max_exp_avg_sq_A"])
            torch.maximum(state["max_exp_avg_sq_B"], exp_avg_sq_B, out=state["max_exp_avg_sq_B"])

            denom_A = (state["max_exp_avg_sq_A"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
            denom_B = (state["max_exp_avg_sq_B"].sqrt() / bias_correction2_sqrt).add_(group['eps'])
        else:
            denom_A = (exp_avg_sq_A.sqrt() / bias_correction2_sqrt).add_(group['eps'])
            denom_B = (exp_avg_sq_B.sqrt() / bias_correction2_sqrt).add_(group['eps'])
            
        if group['weight_decay'] != 0:
            A.mul_(1 - group["weight_decay"] * group["lr"])
            B.mul_(1 - group["weight_decay"] * group["lr"])
            
        A.addcdiv_(exp_avg_A / bias_correction1, denom_A, value=-step_size)
        B.addcdiv_(exp_avg_B / bias_correction1, denom_B, value=-step_size_b)

    def _update_standard_params(self, state, p, group, beta1, beta2):
        grad = p.grad
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_t = state["step"]

        step_t += 1

        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = group['lr'] 

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
        if group['weight_decay'] != 0:
            p.mul_(1 - group["weight_decay"] * group["lr"])
        
        p.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)