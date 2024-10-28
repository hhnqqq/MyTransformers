# @author: haonan he
# @date: 2024-08-21
""" Implements RSLORA"""

from common.lora_modules.lora import *

class LinearWithRSLoRA(LinearWithLoRA):
    def __init__(self,
                lora_config: LoRAConfig):
        super().__init__(lora_config)
        self.lora_scaler = lora_config.lora_scaler / (lora_config.lora_rank**0.5)
