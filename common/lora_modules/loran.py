from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *

class LinearWithLoRAN(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig, freq, amp):
        super().__init__(lora_config)
        self.freq = freq
        self.amp = amp

    def sinter(self, x):
        return self.amp * torch.sin(self.freq * x) * x + x
    
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # If self.run_lora_in_fp32, then the dtype of lora_result will be fp32.
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        
        lora_result = self.sinter(F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)).to(result.dtype)
        return result + self.lora_scaler * lora_result