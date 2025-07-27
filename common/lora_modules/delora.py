# @author: haonan he
# @date: 2025-07-11
"""
Implementaion of DeLoRA: Decoupling Angles and Strength in Low-rank Adaptation [ICLR 2025]
Paper link: https://arxiv.org/abs/2503.18225
Code reference: https://github.com/ExplainableML/DeLoRA/blob/main/peft/src/peft/tuners/delora.py

DeLoRA normalizes and scales learnable low-rank matrices. By bounding the distance of the transformation, 
DeLoRA effectively decouples the angular learning from the adaptation strength, 
enhancing robustness without compromising performance.
"""
from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig

class LinearWithDeLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, delora_lambda):
        """
        Initialize the LinearWithDELoRA layer.

        Args:
            lora_config: General configuration of LoRA and its variants
            delora_lambda: The hyperparameter of delora that controlling the magnitude of lora weights.
        """
        super().__init__(lora_config)
        self.delora_lambda = delora_lambda

    def init_lora_weights(self):
        super().init_lora_weights()
        self.delora_lambda = nn.Parameter(torch.full((1,), self.delora_lambda))
        # Compute this before manipulate the pre-trained weight.
        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)
        self.weight.data = self.weight.data - self._compute_lora_weight()

    def _compute_lora_weight(self):
        # Get weights
        dtype = self._get_lora_dtype()
        device = self.weight.device
        weight_a = self.weight_a.to(dtype).to(device)
        weight_b = self.weight_b.to(dtype).to(device)
        delora_lambda = self.delora_lambda.to(dtype).to(device)

        # Get norms
        weight_a_norm = weight_a.norm(dim=1)  # shape: [rank]
        weight_b_norm = weight_b.norm(dim=0)  # shape: [rank]

        # Compute diagonal scaling factors (avoid constructing full diag matrix)
        diag_values = delora_lambda / (self.lora_rank * weight_a_norm * weight_b_norm)  # shape: [rank]

        # Optimized computation: (weight_b @ diag_values) @ weight_a
        lora_weight = torch.matmul(weight_b * diag_values, weight_a)  # equivalent to weight_b @ diag @ weight_a
        lora_weight.mul_(self.Wnorm.to(device))

        return lora_weight.to(self.weight.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if not self.disable_lora and self.has_lora_weights:
            # merge the weight and low-rank weight.
            # skip a individual forward pass for low-rank weight.
            weight = weight + self._compute_lora_weight()
        return F.linear(x, weight, self.bias)
    
    @property
    def has_lora_weights(self):
        has_lambda = getattr(self, "delora_lambda", None) is not None
        return has_lambda and super().has_lora_weights