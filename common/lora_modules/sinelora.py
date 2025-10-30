from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig

class LinearWithSineLoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, freq):
        super().__init__(lora_config)
        self.freq = freq
        self.lora_scaler = 1 / self.out_features**0.5

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # If self.run_lora_in_fp32, then the dtype of lora_result will be fp32.
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        
        lora_result = torch.sin(self.freq * F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b).to(result.dtype))
        return result + self.lora_scaler * lora_result