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
    F = torch.linalg.solve(U, torch.matmul((C + 0j), V))
    W = R[..., :, None] - S[..., None, :]
    Y = F / W
    X = torch.matmul(torch.matmul(U[...,:n,:n], Y[...,:n,:m]), torch.linalg.inv(V)[...,:m,:m])
    return X.real if all(torch.isreal(x.flatten()[0]) 
                for x in [A, B, C]) else X

class LoRAProAdamW(Optimizer):
    def __init__(
        self,
        named_params: List[Tuple[str, torch.Tensor]],
        lr: Union[float, torch.Tensor] = 1e-3,
        lora_scaler: float = 2.,
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
        self.named_param_dtype = {}
        
        if not isinstance(named_params, list):
            named_params = [named_params]
        # Process named_params into param groups
        params = []

        for named_params_group in named_params:
            param_group = {
                'params': [],
                'params_fp32': [],
                'names': [],
                'grads': [],
                'lr': named_params_group.get('lr', lr),
            }

            for name, param in named_params_group['params']:
                param_group['params'].append(param)
                param_group['params_fp32'].append(param.detach().clone().float())
                param_group['names'].append(name)
                self.named_param_dtype[name] = param.dtype
                
            params.append(param_group)
        
        defaults = dict(
            lr=lr,
            lora_scaler=lora_scaler,
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
            for name, param_fp32, param in zip(group["names"], group["params_fp32"], group["params"]):
                param.data.copy_(param_fp32.data)

        return loss

    def _update_group_params(self, group):
        beta1, beta2 = group["betas"]
        lora_scaler = group["lora_scaler"]

        param_dict, grad_dict = {}, {}
        for i in range(len(group['params'])):
            # Ensure p.data.dtype is torch.float32
            param = group['params'][i]
            param_fp32 = group['params_fp32'][i]
            grad = param.grad
            name = group['names'][i]
            if grad is None:
                continue
            lora_weight_name = find_lora_names(name)
            if lora_weight_name:
                base_name = name[: name.find(lora_weight_name)]
                param_dict[lora_weight_name] = param_fp32
                grad_dict[lora_weight_name] = grad
                if len(param_dict.keys()) == 1:
                    continue
                elif len(param_dict.keys()) == 2:
                    name = base_name + 'lora'
            else:
                name = name
            
            state = self.state[name]
        
            if len(state) == 0:
                self._initialize_state(state, param_dict, param_fp32, group)

            if len(param_dict.keys()) == 2:
                self._update_lora_params(state, param_dict, grad_dict, group, lora_scaler)
                param_dict = {}
                grad_dict = {}
            else:
                self._update_standard_params(state, param_fp32, grad, group, beta1, beta2)


    def _initialize_state(self, state, param_dict, p, group):
        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
        # Ensure optimizer states in torch.float32.
        if len(param_dict.keys()) == 2:
            self._initialize_lora_state(state, param_dict, p.device, torch.float32, group["amsgrad"])
        else:
            self._initialize_standard_state(state, p.shape, p.device, torch.flaot32, group["amsgrad"])

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
    
    def _update_lora_params(self, state, param_dict, grad_dict, group, lora_scaler):
        A = param_dict['weight_a']
        B = param_dict['weight_b']
        lora_rank, in_features = A.shape
        out_features, _ = B.shape
        grad_A_orin_fp32 = grad_dict['weight_a'].to(torch.float32)
        grad_B_orin_fp32 = grad_dict['weight_b'].to(torch.float32)

        delta = 1e-8
        AA_T = torch.matmul(A, A.T)
        B_TB = torch.matmul(B.T, B)
        AA_T_inv = torch.linalg.pinv(AA_T + delta * torch.eye(lora_rank).to(A.device)).to(A.dtype)
        B_TB_inv = torch.linalg.pinv(B_TB + delta * torch.eye(lora_rank).to(A.device)).to(A.dtype)

        X = self._compute_X(group, B, A, lora_scaler, grad_A_orin_fp32, grad_B_orin_fp32, B_TB_inv, AA_T, B_TB).to(B.device).to(B.dtype)

        # [r,r], [r, d] -> [r, d]
        B_TB_inv_B_T = torch.matmul(B_TB_inv, B.T)
        I_minus_BBT_inv = torch.eye(out_features, device=B.device, dtype=B.dtype) - torch.matmul(B, B_TB_inv_B_T)
        
        grad_scale = (1 / lora_scaler ** 2)
        grad_A_fp32 = grad_scale * torch.matmul(B_TB_inv, grad_A_orin_fp32) + torch.matmul(X, A)
        grad_B_fp32 = grad_scale * (torch.matmul(I_minus_BBT_inv, torch.matmul(grad_B_orin_fp32, AA_T_inv))) - torch.matmul(B, X)

        exp_avg_A = state["exp_avg_A"]
        exp_avg_sq_A = state["exp_avg_sq_A"]
        
        exp_avg_B = state["exp_avg_B"]
        exp_avg_sq_B = state["exp_avg_sq_B"]

        step_t = state["step"]

        step_t += 1

        exp_avg_A.lerp_(grad_A_fp32, 1 - group["betas"][0])
        exp_avg_B.lerp_(grad_B_fp32, 1 - group["betas"][0])
        exp_avg_sq_A.mul_(group["betas"][1]).addcmul_(grad_A_fp32, grad_A_fp32.conj(), value=1 - group["betas"][1])
        exp_avg_sq_B.mul_(group["betas"][1]).addcmul_(grad_B_fp32, grad_B_fp32.conj(), value=1 - group["betas"][1])

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

    def _compute_X(self, group, B, A, lora_scaler, grad_A_orin_fp32, grad_B_orin_fp32, B_TB_inv, AA_T, B_TB):
        if group['X_mode'] == "sylvester":
            return solve_sylvester(B_TB, AA_T, -(1 / lora_scaler ** 2) * torch.matmul(torch.matmul(B_TB_inv, grad_A_orin_fp32), A.T))
        elif group['X_mode'] == "symmetry":
            return -0.5 * (1 / lora_scaler ** 2) * torch.matmul(torch.matmul(B_TB_inv, B.T), torch.matmul(grad_B_orin_fp32, AA_T))
        else:
            return torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0]))

    def _update_standard_params(self, state, param, grad, group, beta1, beta2):
        exp_avg = state["exp_avg"]
        exp_avg_sq = state["exp_avg_sq"]
        step_t = state["step"]

        step_t += 1

        grad_fp32 = grad.float()
        exp_avg.lerp_(grad_fp32, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad_fp32, grad_fp32.conj(), value=1 - beta2)

        step = _get_value(step_t)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        step_size = group['lr'] 

        bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)
        denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(group['eps'])
        if group['weight_decay'] != 0:
            param.mul_(1 - group["weight_decay"] * group["lr"])
        
        param.addcdiv_(exp_avg / bias_correction1, denom, value=-step_size)
