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
        
        self.previous_lora_weights = nn.ModuleDict()
        self.update_ratio = update_ratio

    def update_pretrained_weight(self):
        if not self.previous_lora_weights:
            self.previous_lora_weights['A'] = self.weight_a.clone().detach()
            self.previous_lora_weights['B'] = self.weight_b.clone().detach()
        else:
            delta_lora_weight = self._compute_delta_lora_weight()
            
            self.previous_lora_weights['A'] = self.weight_a.clone().detach()
            self.previous_lora_weights['B'] = self.weight_b.clone().detach()
            
            self.weight.data += self.update_ratio * delta_lora_weight

    def _compute_delta_lora_weight(self):
        A = self.weight_a.to(self._get_lora_dtype()) - self.previous_lora_weights['A'].to(self._get_lora_dtype())
        B = self.weight_b.to(self._get_lora_dtype()) - self.previous_lora_weights['B'].to(self._get_lora_dtype())
        return (self.lora_scaler * torch.matmul(A, B)).to(self.weight.dtype)