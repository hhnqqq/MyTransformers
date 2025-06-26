"""
Implementation for Non-Zero LoRA
Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics [ICML 2025]
paper: https://arxiv.org/pdf/2505.23194
code: https://github.com/Leopold1423/non_zero_lora-icml25/blob/icml25-cr/peft/src/peft/tuners/lora.py
"""
from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig

class LinearWithNZLoRA(LinearWithLoRA):
    
    def __init__(self, lora_config: LoRAConfig, init_scale_a, init_scale_b, reset_weight):
        super().__init__(lora_config)
        self.init_scale_a = init_scale_a
        self.init_scale_b = init_scale_b
        self.reset_weight = reset_weight
        
    def init_lora_weights(self):
        super().init_lora_weights()
        if self.reset_weight:
            self.weight_a.data = self.weight_a.data.to(self.weight.device)
            self.weight_b.data = self.weight_b.data.to(self.weight.device)
            lora_weight = self._compute_lora_weight().to(self.weight.dtype)
            self.weight.data = self.weight.data - lora_weight

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        # kaiming uniform
        init_scales = {
            'weight_a': self.init_scale_a,
            'weight_b': self.init_scale_b
        }
        
        try:
            init_scale = init_scales[weight_name]
        except KeyError:
            raise RuntimeError(f'Unrecognized weight_name "{weight_name}" for NZLoRA initialization.'
                               'This should never happend, please report to us!')
        
        bound = (init_scale / self.in_features) ** 0.5
        return {
            'method': 'uniform',
            'a': -bound,
            'b': bound
        }