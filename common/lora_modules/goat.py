# @author: Jingqi Ye
# @date: 2025-06-05
from common.lora_modules.lora import *
import torch
import math

class LinearWithGOAT(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig,
                scalling_type: str = 'lora',
                init_type: str = 'svd',
                num_experts: int = 2,
                top_k: int = 2,
                rho: float = 10, 
                eta: float = 1,
                init_cof: float = 1.0):
        """
        Initialize the LinearWithGOAT layer.

        Args:
            lora_config (LoRAConfig): Configuration for LoRA, including in_features, out_features,
                                       lora_rank, lora_scaler, lora_dropout, quant, and weight initialization methods.
        """
        super().__init__(lora_config)
        if self.quant:
            print(f'Currently GOAT is incompatible with quant, skipped quant')
        self.num_experts = num_experts
        self.top_k = top_k
        self.lora_scaler = lora_config.lora_scaler
        self.scaling_type = scalling_type
        assert self.scaling_type in ['lora', 'rslora', 'goat'], f"Invalid scaling type: {self.scaling_type}. Choose from ['lora', 'rslora', 'goat']."
        self.init_type = init_type
        assert self.init_type in ['goat', 'goat-mini', 'vanilla'], f"Invalid initialization type: {self.init_type}. Choose from ['svd', 'goat-mini', 'vanilla']."

        self.rho = rho
        self.eta = eta
        self.init_cof = init_cof
        # TODO
        # self.n_iters = fast_svd_n_iters
        # self.fast_svd = fast_svd_n_iters > 2
        self.lora_dropout = lora_config.lora_dropout
        self.layer_loss = None

    def _prepare_for_init(self):
        self.lora_rank = self.lora_rank // self.num_experts
        assert self.lora_rank > 0, "LoRA rank must be greater than 0."

        if self.lora_dropout:
            self.dropout = nn.ModuleList([nn.Dropout(self.lora_dropout) for _ in range(self.num_experts)])
        else:
            self.dropout = nn.ModuleList([nn.Identity() for _ in range(self.num_experts)])
        
        if self.scaling_type == 'goat':
            self.scaling = math.sqrt(3*self.eta*self.in_features / self.lora_rank)
        elif self.scaling_type == 'rslora':
            self.scaling = self.lora_scaler / math.sqrt(self.lora_rank)
        else:
            self.scaling = self.lora_scaler / self.lora_rank

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype

        self._prepare_for_init()

        # Initialize the router and gating parameters.
        self.lora_gating = nn.Linear(self.in_features, self.num_experts, bias=False, dtype=self._get_lora_dtype())
        print(f"init_lora_weights lora_gating dtype: {self.lora_gating.weight.dtype}, lora_gating weight shape: {self.lora_gating.weight.shape}")

        self.weight_a, self.weight_b = nn.ParameterList(), nn.ParameterList()
        # For inference initialization and skip the SVD process
        if self.init_type == 'vanilla':
            self.weight_a.extend([nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=False) for _ in range(self.num_experts)])
            self.weight_b.extend([nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=False) for _ in range(self.num_experts)])
            self._init_weight('weight_a')
            self._init_weight('weight_b')
        else:
            weight = self.weight.to(torch.float32)
            V, S, Uh = torch.linalg.svd(weight.data, full_matrices=False)
            Vlen = V.shape[-1]//self.num_experts
            Mlen = self.lora_rank
            if self.init_type == 'goat':
                print(f"GOAT initialization with {self.num_experts} experts and rank {self.lora_rank}.")
                V_piece = [V[:, i*Vlen:i*Vlen+Mlen] for i in range(self.num_experts)]
                S_piece = [S[i*Vlen:i*Vlen+Mlen] for i in range(self.num_experts)]
                U_piece = [Uh[i*Vlen:i*Vlen+Mlen] for i in range(self.num_experts)]
                # Vr = torch.cat(V_piece, dim=1)
                # Sr = torch.cat(S_piece)
                # Uhr = torch.cat(U_piece, dim=0)
                # Sr /= self.scaling * self.rho
            elif self.init_type == "goat_mini":
                V_piece = [V[:, (i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(self.num_experts)]
                S_piece = [S[(i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(self.num_experts)]
                U_piece = [Uh[(i+1)*Vlen-Mlen:(i+1)*Vlen] for i in range(self.num_experts)]
                
            Vr = torch.cat(V_piece, dim=1)
            Sr = torch.cat(S_piece)
            Uhr = torch.cat(U_piece, dim=0)
            Sr /= self.scaling * self.rho
            lora_A = torch.diag(torch.sqrt(Sr)) @ Uhr
            lora_B = Vr @ torch.diag(torch.sqrt(Sr))

            for i in range(self.num_experts):
                self.weight_a.append(lora_A[i * self.lora_rank : (i+1) * self.lora_rank, :].contiguous().to(dtype=dtype))
                self.weight_b.append(lora_B[:, i * self.lora_rank : (i+1) * self.lora_rank].contiguous().to(dtype=dtype))
            self.weight.data = (weight - self.init_cof * self.scaling * lora_B @ lora_A).to(weight_dtype)

    # For inference initialization and skip the SVD process
    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # x = x.to(self._get_lora_dtype())
        flatttened_x = x.view(-1, x.shape[-1]).to(self._get_lora_dtype()) # [batch_size * seq_len, in_features]
        print(f"flatttened_x dtype:{flatttened_x.dtype}, lora_gating dtype:{self.lora_gating.weight.dtype}")
        self.lora_gating.weight.data = self.lora_gating.weight.data.to(flatttened_x.dtype)
        gate_logits = F.softmax(self.lora_gating(flatttened_x), dim=-1)  # [batch_size * seq_len, num_experts]
        # Select top-k experts based on gate logits
        weights, selected_experts = torch.topk(input=gate_logits, k=self.top_k, dim=-1) # [batch_size * seq_len, top_k]

        # Renormalize weights
        weights = weights / (torch.sum(weights, dim=-1, keepdim=True, dtype=self._get_lora_dtype()) + 1e-8)  # Avoid division by zero
        lora_results = torch.zeros_like(result.view(-1, self.out_features), dtype=self._get_lora_dtype())
        # Moe forward pass
        for i in range(self.num_experts):
            token_idx, expert_idx = torch.where(selected_experts == i)
            if len(token_idx) == 0:
                continue
            lora_results[token_idx] += \
                weights[token_idx, expert_idx].unsqueeze(-1) * F.linear(F.linear(self.dropout[i](flatttened_x[token_idx]), self.weight_a[i].to(dtype=self._get_lora_dtype())), self.weight_b[i].to(dtype=self._get_lora_dtype()))
        
        lora_results = lora_results.view(*result.shape).to(result.dtype)
        self.get_layer_loss(gate_logits, selected_experts)
        return result + self.scaling * lora_results
    
    def get_layer_loss(self, gate_logits: torch.Tensor, selected_experts: torch.Tensor) -> torch.Tensor:
        num_tokens = gate_logits.shape[0]
        num_experts = self.num_experts
        expert_counts = torch.bincount(selected_experts.view(-1), minlength=num_experts)
        expert_fractions = expert_counts.float() / num_tokens
        expert_probs = torch.sum(gate_logits, dim=0) / num_tokens
        layer_loss = num_experts * torch.sum(expert_fractions * expert_probs)
        return layer_loss
