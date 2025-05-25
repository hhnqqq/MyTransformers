# @author: haonan he
# @date: 2024-08-21
""" Implements MELORA"""

from common.lora_modules.lora import *

class LinearWithLoRAMoE(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        lora_moe_n_experts: int = 2,
        lora_moe_top_k: int = 2):
        """
        Initialize the LinearWithMosLoRA layer.

        Args:
           lora_moe_n_experts int: Number of groups of LoRA weight.

        Note:
            For detailed explanations of in_features, out_features, lora_rank, lora_scaler, 
            lora_dropout, quant, weight_a_init_method, and weight_b_init_method, 
            please refer to the parent class LinearWithLoRA.
        """
        self.lora_moe_n_experts = lora_moe_n_experts
        self.moe_top_k = lora_moe_top_k
        super().__init__(lora_config)
        if lora_config.quant:
            print(f'Currently LoRAMoE is incompatible with quant, skipped quant')

    @property
    def requires_gate(self):
        return not(self.lora_moe_n_experts<=self.moe_top_k)
    
    def init_lora_weights(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a, self.weight_b =nn.ParameterList(), nn.ParameterList()
        if self.requires_gate:
            self.gate = nn.Parameter(torch.rand((self.moe_top_k, self.in_features), dtype=dtype), requires_grad=requires_grad)
        for _ in range(self.lora_moe_n_experts):
            expert_weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
            expert_weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            self.weight_a.append(expert_weight_a)
            self.weight_b.append(expert_weight_b)
        self._init_weight('weight_a')
        self._init_weight('weight_b')

    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        lora_results = torch.zeros(x.shape[0]*x.shape[1], self.out_features, device=x.device, dtype=result.dtype)
        origin_shape = x.shape
        x = x.view(-1, x.shape[-1])
        if self.requires_gate:
            # Shape: [bsz*seq_len, out] --> [bsz*seq_len, top_k]
            gate_logits = F.linear(x, self.gate.to(self._get_lora_dtype()))
            weights, selected_experts = torch.topk(
                gate_logits, self.moe_top_k
            )
            weights = torch.nn.functional.softmax(weights, dim=1, dtype=torch.float).to(
                x.dtype
            )
            for i, (expert_weight_a, expert_weight_b) in enumerate(zip(self.weight_a, self.weight_b)):
                expert_weight_a = expert_weight_a.to(self._get_lora_dtype())
                expert_weight_b = expert_weight_b.to(self._get_lora_dtype())
                batch_idx, nth_expert = torch.where(selected_experts == i)
                lora_results[batch_idx] += weights[batch_idx, nth_expert, None] * F.linear(F.linear(self.lora_dropout(x[batch_idx]), expert_weight_a), expert_weight_b).to(result.dtype)
        else:
            for i, (expert_weight_a, expert_weight_b) in enumerate(zip(self.weight_a, self.weight_b)):
                expert_weight_a = expert_weight_a.to(self._get_lora_dtype())
                expert_weight_b = expert_weight_b.to(self._get_lora_dtype())
                lora_results += F.linear(F.linear(self.lora_dropout(x), expert_weight_a), expert_weight_b).to(result.dtype)
            lora_results /= self.lora_moe_n_experts
        return result + self.lora_scaler * lora_results.reshape(*origin_shape[:2], self.out_features)
    
    def _compute_lora_weight(self):
        lora_weight = torch.zeros((self.in_features, self.out_features))
        if not self.requires_gate:
            for i, (expert_weight_a, expert_weight_b) in enumerate(zip(self.weight_a, self.weight_b)):
                expert_weight_a = expert_weight_a.to(self._get_lora_dtype())
                expert_weight_b = expert_weight_b.to(self._get_lora_dtype())
                lora_weight += self.lora_scaler * torch.matmul(expert_weight_b, expert_weight_a)
            return lora_weight.to(self.weight.dtype)
        else:
            raise ValueError('LoRAMoE is not competible with _compute_lora_weight when requires_gate==True')
        