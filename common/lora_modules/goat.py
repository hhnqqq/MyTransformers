# @author: Jingqi Ye (modified by: haonan he)
# @date: 2025-06-05
"""
Implementation of GOAT [ICML 2025]
(Make LoRA Great Again: Boosting LoRA with Adaptive Singular Values and Mixture-of-Experts Optimization Alignment)
Paper link: https://arxiv.org/abs/2502.16894
"""
from common.lora_modules.lora import *
import torch
import math

class LinearWithGOAT(LinearWithLoRA):
    def __init__(self, 
        lora_config: LoRAConfig,
        scaling_type: str = None,
        init_type: str = 'goat',
        n_experts: int = 2,
        top_k: int = 2,
        rho: float = 10, 
        eta: float = 1,
        init_cof: float = 1.0):
        """
        Initialize the LinearWithGOAT layer.

        Args:
            lora_config (LoRAConfig): Configuration for LoRA
            scaling_type (str): Lora scaler setting.
            init_tyoe (str): Init methods for low-rank weights.
            n_experts (int): Number of LoRA experts.
            top_k (int): Number of activated LoRA experts per token.
            rho (float): Hyperparameter of GOAT controlling the strength of weight_a.
            eta (float): Hyperparameter of GOAT controlling the strength of lora_scaler.
            init_cof (float): Hyperparameter of GOAT controlling the strength of manipulation of pre-trained weight. 
        """
        super().__init__(lora_config)
        self.n_experts = n_experts
        self.top_k = top_k
        self.init_type = init_type
        self.lora_rank = math.ceil(lora_config.lora_rank / self.n_experts)
        

        if scaling_type == 'goat':
            self.lora_scaler = math.sqrt(3*self.eta*self.in_features / self.lora_rank)
        elif scaling_type == 'rslora':
            self.lora_scaler = math.ceil(lora_config.lora_scaler / self.lora_rank) / math.sqrt(self.lora_rank)
        else:
            self.lora_scaler = math.ceil(lora_config.lora_scaler / self.lora_rank) / self.lora_rank

        self.rho = rho
        self.eta = eta
        self.init_cof = init_cof
        self.layer_loss = None
        if lora_config.lora_dropout:
            self.lora_dropout = nn.ModuleList([nn.Dropout(lora_config.lora_dropout) for _ in range(self.n_experts)])
        else:
            self.lora_dropout = nn.ModuleList([nn.Identity() for _ in range(self.n_experts)])

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype
        
        # Initialize the router and gating parameters
        self.gate = nn.Parameter(
            torch.rand((self.n_experts, self.in_features), 
            dtype=dtype,
            requires_grad=True
        ))

        self.weight_a = nn.ParameterList()
        self.weight_b = nn.ParameterList()

        if self.init_type == 'vanilla':
            # Vanilla initialization
            for _ in range(self.n_experts):
                self.weight_a.append(nn.Parameter(
                    torch.empty((self.lora_rank, self.in_features), dtype=dtype),
                    requires_grad=True
                ))
                self.weight_b.append(nn.Parameter(
                    torch.zeros((self.out_features, self.lora_rank), dtype=dtype),
                    requires_grad=True
                ))
            self._init_weight('weight_a')
            self._init_weight('weight_b')
        else:
            # SVD-based initialization
            weight = self.weight.to(torch.float32)
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vlen = V.shape[-1] // self.n_experts
            
            # Select slices based on initialization type
            if self.init_type == 'goat':
                slices = [slice(i*Vlen, i*Vlen+self.lora_rank) for i in range(self.n_experts)]
            elif self.init_type == 'goat_mini':
                slices = [slice((i+1)*Vlen-self.lora_rank, (i+1)*Vlen) for i in range(self.n_experts)]
            
            # Process each expert's components
            V_pieces = [V[:, sl] for sl in slices]
            S_pieces = [S[sl] for sl in slices]
            U_pieces = [Uh[sl] for sl in slices]
            
            # Combine and normalize components
            Vr = torch.cat(V_pieces, dim=1)
            Sr = torch.cat(S_pieces) / (self.lora_scaler * self.rho)
            Uhr = torch.cat(U_pieces, dim=0)
            
            # Compute A and B matrices
            sqrt_Sr = torch.sqrt(Sr)
            A = torch.diag(sqrt_Sr) @ Uhr
            B = Vr @ torch.diag(sqrt_Sr)
            
            # Assign expert weights
            for i in range(self.n_experts):
                a_slice = A[i*self.lora_rank:(i+1)*self.lora_rank, :]
                b_slice = B[:, i*self.lora_rank:(i+1)*self.lora_rank]
                self.weight_a.append(a_slice.contiguous().to(dtype=dtype))
                self.weight_b.append(b_slice.contiguous().to(dtype=dtype))
            
            # Update original weight
            self.weight.data = (weight - self.init_cof * self.lora_scaler * (B @ A)).to(weight_dtype)

    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        flatttened_x = x.view(-1, x.shape[-1]).to(self._get_lora_dtype()) # [batch_size * seq_len, in_features]
        # print(f"flatttened_x dtype:{flatttened_x.dtype}, lora_gating dtype:{self.lora_gating.weight.dtype}")
        gate_logits = F.softmax(F.linear(flatttened_x, self.gate.to(self._get_lora_dtype())), dim=-1)  # [batch_size * seq_len, n_experts]
        # Select top-k experts based on gate logits
        weights, selected_experts = torch.topk(input=gate_logits, k=self.top_k, dim=-1) # [batch_size * seq_len, top_k]

        # Renormalize weights
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True, dtype=self._get_lora_dtype()) + 1e-8)  # Avoid division by zero
        lora_results = torch.zeros_like(result.view(-1, self.out_features), dtype=self._get_lora_dtype())
        # Moe forward pass
        for i in range(self.n_experts):
            token_idx, expert_idx = torch.where(selected_experts == i)
            if len(token_idx) == 0:
                continue
            lora_results[token_idx] += \
                weights[token_idx, expert_idx].unsqueeze(-1) * F.linear(F.linear(self.lora_dropout[i](flatttened_x[token_idx]), self.weight_a[i].to(dtype=self._get_lora_dtype())), self.weight_b[i].to(dtype=self._get_lora_dtype()))
        
        lora_results = lora_results.view(*result.shape).to(result.dtype)
        self.get_layer_loss(gate_logits, selected_experts)
        return result + self.lora_scaler * lora_results
    
    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        num_tokens = gate_logits.shape[0]
        n_experts = self.n_experts
        expert_counts = torch.bincount(selected_experts.view(-1), minlength=n_experts)
        expert_fractions = expert_counts.float() / num_tokens
        expert_probs = torch.sum(gate_logits, dim=0) / num_tokens
        self.layer_loss = n_experts * torch.sum(expert_fractions * expert_probs)
