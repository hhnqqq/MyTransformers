from common.lora_modules.lora import *

class LinearWithDeltaLoRA(LinearWithLoRA):
    def __init__(self,
                 lora_config: LoRAConfig,
                 update_ratio: float = 1e-6):
        super().__init__(lora_config)
        if lora_config.lora_dropout is not None:
            raise ValueError('DeltaLoRA is not compatible with dropout.')
        
        self.previous_lora_weights = nn.ModuleDict()
        self.update_ratio = update_ratio

    def update_pretrained_weight(self):
        if not self.previous_lora_weights:
            self.previous_lora_weights['A'] = self.weight_a.clone().detach()
            self.previous_lora_weights['B'] = self.weight_b.clone().detach()
        else:
            delta_lora_weight = self._compute_lora_weight() - self._compute_previous_lora_weight()
            
            self.previous_lora_weights['A'] = self.weight_a.clone().detach()
            self.previous_lora_weights['B'] = self.weight_b.clone().detach()
            
            self.weight.data += self.update_ratio * delta_lora_weight

    def _compute_previous_lora_weight(self):
        A = self.previous_lora_weights['A'].to(self._get_lora_dtype())
        B = self.previous_lora_weights['B'].to(self._get_lora_dtype())
        return (self.lora_scaler * torch.matmul(A, B)).to(self.weight.dtype)