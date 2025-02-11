from common.lora_modules.lora import *

class LinearWithPLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        plora_momentum: Optional[float] = 0.1,
    ):
        self.momentum = plora_momentum
        super().__init__(lora_config)

    def merge_and_reset_with_momentum(self):
        lora_result: torch.Tensor = self._compute_lora_weight()
        lora_result_with_momentum: torch.Tensor = self.momentum * lora_result
        self.weight_a.data = (1-self.momentum) * self.weight_a.data.detach()
        self.weight_b.data = (1-self.momentum) * self.weight_b.data.detach()
        self.weight.data = self.weight.data + lora_result_with_momentum.detach()
