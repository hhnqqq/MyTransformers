"""
 Algorithm : Delta-LoRA
 Input: Learning rate η; weight decay β; total training iterations T; low rank r; scale factor α; start
 steps K; update ratio λ.
 Ais initialized by Kaiming Initialization, B = 0 and W is initialized with pre-trained weights.
 for t = 0,...,T -1 do
    Sample a mini-batch and compute gradients for {A,B} in each Delta-LoRA module.
    Update the first and second moments maintained by the optimizer with the computed gradients,
and get the normalized gradients gA and gB.
    A(t+1) ← A(t) -ηgA -ηβA(t)
    B(t+1) ← B(t) -ηgB -ηβB(t)
    if t > K do
        W(t+1) ← W(t) +λ· α
        r ·(A(t+1)B(t+1) -A(t)B(t))
    end if
 end for
 Output: the fine-tuned parameters {W(T),A(T),B(T)}

We use update ratio λ=2 and set start steps K=500 for Delta-LoRA.
"""
from common.lora_modules.lora import *

class LinearWithDeltaLoRA(LinearWithLoRA):
    def __init__(self,
                 lora_config: LoRAConfig,
                 update_ratio: float = 2):
        super().__init__(lora_config)
        if lora_config.lora_dropout is not None:
            raise ValueError('DeltaLoRA is not compatible with dropout.')
        
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