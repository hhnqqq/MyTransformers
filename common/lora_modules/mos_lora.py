# @author: haonan he
# @date: 2024-08-21
""" Implements MosLORA"""

from common.lora_modules.lora import *

class LinearWithMosLoRA(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        quant: bool = False,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None):
        self.weight_ab_mixer_init_method = weight_ab_mixer_init_method
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         quant,
                         weight_a_init_method,
                         weight_b_init_method)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
        weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
        weight_a = torch.matmul(weight_ab_mixer, weight_a)
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)

        return result + self.lora_scaler * lora_result
    
    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
            # When using vanilla lora, the ab mixer is a identical matrix

            weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            return lora_weight
        
    def _init_lora_weights(self):
        super()._init_lora_weights()
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        if self.quant:
            self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
        self._init_weight('weight_ab_mixer')