# @author: haonan he
# @date: 2024-08-21
""" Un-official implements MosLORA.
Mixture-of-Subspaces in Low-Rank Adaptation (https://arxiv.org/pdf/2406.11909)
MoSLoRA consistently outperforms LoRA on tasks in different modalities, 
including commonsense reasoning, visual instruction tuning, 
and subjectdriven text-to-image generation, 
demonstrating its effectiveness and robustness."""

from common.lora_modules.lora import *

class LinearWithMosLoRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        weight_ab_mixer_init_method: Optional[str] = None):
        """
        Initialize the LinearWithMosLoRA layer.

        Args:
            weight_ab_mixer_init_method (str, optional): The init method for weight_ax_mixer.

        Note:
            For detailed explanations of in_features, out_features, lora_rank, lora_scaler, 
            lora_dropout, quant, weight_a_init_method, and weight_b_init_method, 
            please refer to the parent class LinearWithLoRA.
        """
        self.weight_ab_mixer_init_method = weight_ab_mixer_init_method
        super().__init__(lora_config)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
        weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
        # weight_a = torch.matmul(weight_ab_mixer, weight_a)
        lora_result = F.linear(F.linear(F.linear(self.lora_dropout(x), weight_a), weight_ab_mixer), weight_b)
        return result + self.lora_scaler * lora_result
    
    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
            # When using vanilla lora, the ab mixer is a identical matrix

            weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            return lora_weight
        
    def _init_lora_weights(self):
        super()._init_lora_weights()
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        if self.quant:
            self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
        self._init_weight('weight_ab_mixer')

    def merge_and_reset(self, new_rank: Optional[int] = None):
        super().merge_and_reset(new_rank=new_rank)
        if new_rank is None:
            self._init_weight('weight_ab_mixer')
            if self.quant:
                self.weight_ab_scaler = nn.Parameter(torch.Tensor(self.lora_rank))

    @property
    def weight_ab_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_ab_scaler", None)
    
    def _del_lora(self):
        super()._del_lora()
        delattr(self, "weight_ab_mixer")

    @property
    def has_lora_weights(self):
        has_ab_mixer = hasattr(self, 'weight_ab_mixer') and self.weight_ab_mixer is not None
        return has_ab_mixer and super().has_lora_weights()
    