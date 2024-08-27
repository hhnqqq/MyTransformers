# @author: haonan he
# @date: 2024-08-21
""" Implements MosLORA"""

from typing import Union
from common.lora_modules.lora import *

class LinearWithPLoRA(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None):
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         quant,
                         plora_steps,
                         weight_a_init_method,
                         weight_b_init_method)
        self.plora_steps = plora_steps
        self.plora_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every plora stage, we merge the origin lora weight and reset new lora weight.:
        self.plora_counter += 1
        if self.plora_counter == self.plora_steps:
            self.merge_and_reset()
            self.plora_counter = 0

        return super().forward(x)