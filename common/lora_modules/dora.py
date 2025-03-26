"""
Implementation of DoRA

Below is the specific method from the dora paper:
This analysis examines the updates in
both magnitude and direction of the LoRA and FT weights
relative to the pre-trained weights to reveal the fundamental
differences in the learning behaviors of both. The weight
decomposition of W ∈ Rdxk can be formulated as:
 W =m x V / ||V ||c = ||W||c x W / ||W||c

where m ∈ R1xk is the magnitude vector, V ∈ Rdxk is the
directional matrix, with ||·||c being the vector-wise norm of
a matrix across each column. This decomposition ensures
that each column of V/||V ||c remains a unit vector, and the
corresponding scalar in m defines the magnitude of each
vector.
"""
from common.lora_modules.lora import *

class LinearWithDoRA(LinearWithLoRA):
    def __init__(self,
                lora_config: LoRAConfig):
        super().__init__(lora_config)
        if lora_config.lora_dropout:
            print(f'Dora is incompatible with lora dropout, skiped lora dropout')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)
        if not self.disable_lora:
            weight = self._apply_dora(weight)
        return F.linear(x, weight, self.bias)
    
    def _apply_dora(self, weight: torch.Tensor) -> torch.Tensor:
        # Make sure that the dtype of weight same as dtype of lora weights.
        lora_weight = self._compute_lora_weight()

        origin_weight_dtype = weight.dtype
        # Compute column-wise L2 norm.
        origin_magnitude: torch.Tensor = torch.linalg.norm(weight.detach(), dim=1).to(lora_weight.dtype)
        
        weight = weight.to(lora_weight.dtype)
        weight = weight + lora_weight * self.lora_scaler
        new_magnitude: torch.Tensor = torch.linalg.norm(weight.detach(), dim=1).to(lora_weight.dtype)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        new_magnitude = new_magnitude.detach()
        origin_magnitude = origin_magnitude.detach()

        # In peft. This should be added on top of the base layer output.
        # result_dora = (mag_norm_scale - 1) * (
        # F.linear(x, transpose(weight, self.fan_in_fan_out))
        # ) + mag_norm_scale * lora_result * scaling
        mag_norm_scale = (origin_magnitude / new_magnitude).view(-1, 1)
        weight = mag_norm_scale * weight
        return weight.to(origin_weight_dtype)

    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            self.weight.data = self._apply_dora(self.weight)
            return True
        return False

