from common.utils import print_rank_0
from common.lora_modules.lora import *

class LinearWithDoRA(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        use_dora: bool = False,
        use_mos_lora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None):
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         use_dora,
                         use_mos_lora,
                         quant,
                         plora_steps,
                         weight_a_init_method,
                         weight_b_init_method,
                         weight_ab_mixer_init_method)
        if lora_dropout:
            print_rank_0(f'Dora is incompatible with lora dropout, skiped lora dropout')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every plora stage, we merge the origin lora weight and reset new lora weight.
        if self.plora:
            self.plora_counter += 1
            if self.plora_counter == self.plora_steps:
                self.merge_and_reset()
                self.plora_counter = 0

        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)

        lora_weight = None
        # If lora attrs are exist, compute the lora weight and plus it to full rank weight
        if self.has_lora_weights:
            lora_weight = self._compute_lora()
            weight = self._apply_dora(weight, lora_weight)

        return F.linear(x, weight)
    
    def _apply_dora(self, weight: torch.Tensor, lora_weight: torch.Tensor) -> torch.Tensor:
        # The magnitude of origin weight on the output dim: [2048,2048] -> [1, 2048].
        m = self.weight.norm(p=2, dim=0, keepdim=True)
        # Origin weight plus lora weight -> new weight. 
        directional_numerator = weight + lora_weight
        # The magnitude of new weight on the output dim. 
        directional_denominator = directional_numerator.norm(p=2, dim=0, keepdim=True)
        # Scale the magnitude of new weight to 1.
        directional_component = directional_numerator / directional_denominator
        # Ensure the new weight's magnitude remains the same as the origin weight.
        return m * directional_component
