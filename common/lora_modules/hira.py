# @author: haonan he
"""
Implementation for HiRA: Parameter-Efficient Hadamard High-Rank Adaptation for LLMs [ICLR 2025]
Paper link: https://openreview.net/forum?id=TwJrTz9cRS
Code reference: https://github.com/hqsiswiliam/hira/blob/master/hira/tuners/lora.py
"""
from common.lora_modules.lora import *

class LinearWithHiRA(LinearWithLoRA):
    def __init__(self,
                lora_config: LoRAConfig):
        super().__init__(lora_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)
        if not self.disable_lora:
            weight = self._apply_hira(weight)
        return F.linear(x, weight, self.bias)
    
    def _apply_hira(self, weight: torch.Tensor) -> torch.Tensor:
        origin_weight_dtype = weight.dtype
        lora_weight = self._compute_lora_weight()
        weight = weight.to(lora_weight.dtype)
        delta_weight = lora_weight * weight
        weight = weight + delta_weight
        return weight.to(origin_weight_dtype)
    
    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            self.weight.data = self._apply_hira(self.weight)
            return True
        return False