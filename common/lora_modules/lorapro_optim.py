"""Implementation of lora-pro 
Code reference: https://github.com/mrflogs/LoRA-Pro/blob/main/peta/optim.py#L3
This implementation only contain adamw implementation
"""
import torch

from typing import cast, Tuple, Union, List

from torch.optim import Optimizer
from scipy.linalg import solve_sylvester
from torch.optim.adamw import adamw

from common.lora_modules.lora import find_lora_names


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
        self.fake_step =  torch.tensor(0.0, dtype=_get_scalar_dtype())
        
        if not isinstance(named_params, list):
            named_params = [named_params]
        # Process named_params into param groups
        params = []

        for named_params_group in named_params:
            param_group = {
                'params': [],
                'params_fp32': [],
                'names': [],
                'lr': 1,
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
            for param_fp32, param in zip(group["params_fp32"], group["params"]):
                param.data.copy_(param_fp32.data)

        return loss

    def _update_group_params(self, group):
        beta1, beta2 = cast(Tuple[float, float], group["betas"])
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
                if group["amsgrad"]:
                    max_exp_avg_sqs=[state["max_exp_avg_sq"]]
                else:
                    max_exp_avg_sqs=[]

                adamw(params=[param_fp32],
                    grads=[grad.to(torch.float32)],
                    exp_avgs=[state["exp_avg"]],
                    exp_avg_sqs=[state["exp_avg_sq"]],
                    max_exp_avg_sqs=max_exp_avg_sqs,
                    state_steps=[state["step"]],
                    amsgrad=group["amsgrad"],
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"])


    def _initialize_state(self, state, param_dict, p, group):
        state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
        # Ensure optimizer states in torch.float32.
        if len(param_dict.keys()) == 2:
            self._initialize_lora_state(state, param_dict, group["amsgrad"])
        else:
            self._initialize_standard_state(state, p, group["amsgrad"])

    def _initialize_lora_state(self, state, param_dict, amsgrad):
        state["exp_avg_A"] = torch.zeros_like(param_dict['weight_a'], memory_format=torch.preserve_format)
        state["exp_avg_B"] = torch.zeros_like(param_dict['weight_b'], memory_format=torch.preserve_format)
        state["exp_avg_sq_A"] = torch.zeros_like(param_dict['weight_a'], memory_format=torch.preserve_format)
        state["exp_avg_sq_B"] = torch.zeros_like(param_dict['weight_b'], memory_format=torch.preserve_format)

        if amsgrad:
            state["max_exp_avg_sq_A"] = torch.zeros_like(param_dict['weight_a'], memory_format=torch.preserve_format)
            state["max_exp_avg_sq_B"] = torch.zeros_like(param_dict['weight_b'], memory_format=torch.preserve_format)

    def _initialize_standard_state(self, state, p, amsgrad):
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        
        if amsgrad:
            state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    
    def _update_lora_params(self, state, param_dict, grad_dict, group, lora_scaler):
        A: torch.Tensor = param_dict['weight_a']
        B: torch.Tensor = param_dict['weight_b']
        lora_rank, _ = A.shape
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

        exp_avg_A: torch.Tensor = state["exp_avg_A"]
        exp_avg_sq_A: torch.Tensor = state["exp_avg_sq_A"]
        
        exp_avg_B: torch.Tensor = state["exp_avg_B"]
        exp_avg_sq_B: torch.Tensor = state["exp_avg_sq_B"]

        if group["amsgrad"]:
            max_exp_avg_sq_A = state["max_exp_avg_sq_A"]
            max_exp_avg_sq_B = state["max_exp_avg_sq_B"]
            max_exp_avg_sqs = [max_exp_avg_sq_A, max_exp_avg_sq_B]
        else:
            max_exp_avg_sqs = []

        
        adamw(params=[A, B],
             grads=[grad_A_fp32, grad_B_fp32],
             exp_avgs=[exp_avg_A, exp_avg_B],
             exp_avg_sqs=[exp_avg_sq_A, exp_avg_sq_B],
             max_exp_avg_sqs=max_exp_avg_sqs,
             state_steps=[state["step"], self.fake_step],
             amsgrad=group["amsgrad"],
             beta1=group["betas"][0],
             beta2=group["betas"][1],
             lr=group["lr"],
             weight_decay=group["weight_decay"],
             eps=group["eps"],
             maximize=group["maximize"],
             foreach=False)
        

    def _compute_X(self, group, B, A, lora_scaler, grad_A_orin_fp32, grad_B_orin_fp32, B_TB_inv, AA_T, B_TB):
        if group['X_mode'] == "sylvester":
            return solve_sylvester(B_TB, AA_T, -(1 / lora_scaler ** 2) * torch.matmul(torch.matmul(B_TB_inv, grad_A_orin_fp32), A.T))
        elif group['X_mode'] == "symmetry":
            return -0.5 * (1 / lora_scaler ** 2) * torch.matmul(torch.matmul(B_TB_inv, B.T), torch.matmul(grad_B_orin_fp32, AA_T))
        else:
            return torch.zeros((B_TB_inv.shape[0], B_TB_inv.shape[0]))

