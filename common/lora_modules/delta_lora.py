# @author: haonan he
"""
Implementation of Delta-LoRA: Fine-Tuning High-Rank Parameters with the Delta of Low-Rank Matrices [arxiv preprint]
Paper Link: https://arxiv.org/abs/2309.02411
Code reference: None

Delta-LoRA not only updates the low-rank matrices $\bA$ and $\bB$, 
but also propagate the learning to the pre-trained weights $\bW$ via updates 
utilizing the delta of the product of two low-rank matrices ($\bA^{(t+1)}\bB^{(t+1)} - \bA^{(t)}\bB^{(t)}$). 
Such a strategy effectively addresses the limitation that the incremental update of low-rank matrices 
is inadequate for learning representations capable for downstream tasks. 
Moreover, as the update of $\bW$ does not need to compute the gradients of $\bW$ and store their momentums, 
Delta-LoRA shares comparable memory requirements and computational costs with LoRA. 
"""
from common.lora_modules.lora import *

class LinearWithDeltaLoRA(LinearWithLoRA):
    def __init__(self,
                 lora_config: LoRAConfig,
                 update_ratio: float = 2):
        super().__init__(lora_config)
        
        self.previous_lora_weights = {}
        self.update_ratio = update_ratio

    def update_pretrained_weight(self):
        if self.previous_lora_weights:
            delta_lora_weight = self._compute_delta_lora_weight()
            self.weight.data += self.update_ratio * delta_lora_weight

        self.previous_lora_weights['A'] = self.weight_a.clone().detach()
        self.previous_lora_weights['B'] = self.weight_b.clone().detach()

    def _compute_delta_lora_weight(self):
        previous_A = self.previous_lora_weights['A'].to(self._get_lora_dtype())
        previous_B = self.previous_lora_weights['B'].to(self._get_lora_dtype())
        previous_AB = self.lora_scaler * torch.matmul(previous_B, previous_A)
        AB = self._compute_lora_weight()
        return AB - previous_AB