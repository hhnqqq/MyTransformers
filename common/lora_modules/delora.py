from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig

class LinearWithDELoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, delora_lambda):
        super().__init__(lora_config)
        self.delora_lambda = delora_lambda

    def init_lora_weights(self):
        super().init_lora_weights()
        self.delora_lambda = nn.Parameter(torch.full((1,), self.delora_lambda))
        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)
        self.weight.data = self.weight.data - self._compute_lora_weight()
        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)

    def _compute_lora_weight(self):
        # Get weights
        dtype = self._get_lora_dtype()
        device = self.weight.device
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(dtype).to(device)
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(dtype).to(device)
        delora_lambda = self.delora_lambda.to(dtype).to(device)

        # Get norms
        weight_a_norm = weight_a.norm(dim=1)  # shape: [rank]
        weight_b_norm = weight_b.norm(dim=0)  # shape: [rank]

        # Compute diagonal scaling factors (avoid constructing full diag matrix)
        diag_values = delora_lambda / (self.lora_rank * weight_a_norm * weight_b_norm)  # shape: [rank]

        # Optimized computation: (weight_b * diag_values) @ weight_a
        lora_weight = torch.matmul(weight_b * diag_values, weight_a)  # equivalent to weight_b @ diag @ weight_a
        lora_weight.mul_(self.Wnorm)

        return lora_weight
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = self.weight.dtype
        weight = self._quantize_weight(self.weight, self.weight_quantizer)
        if not self.disable_lora:
            weight = weight + self._compute_lora_weight().to(origin_dtype)
        return F.linear(x, weight, self.bias)