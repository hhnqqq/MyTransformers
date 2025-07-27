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
            lora_dropout, weight_a_init_method, and weight_b_init_method, 
            please refer to the parent class LinearWithLoRA.
        """
        self.weight_ab_mixer_init_method = weight_ab_mixer_init_method
        super().__init__(lora_config)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        weight_ab_mixer = self.weight_ab_mixer.to(self._get_lora_dtype())
        # weight_a = torch.matmul(weight_ab_mixer, weight_a)
        lora_result = F.linear(F.linear(F.linear(self.lora_dropout(x), weight_a), weight_ab_mixer), weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_result
    
    def _compute_lora_weight(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self.weight_a.to(self._get_lora_dtype())
            weight_b = self.weight_b.to(self._get_lora_dtype())
            weight_ab_mixer = self.weight_ab_mixer.to(self._get_lora_dtype())
            # When using vanilla lora, the ab mixer is a identical matrix

            weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            return lora_weight
        
    def init_lora_weights(self):
        super().init_lora_weights()
        dtype = self._get_lora_dtype()

        self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=True)
        self._init_weight('weight_ab_mixer')

    def merge_and_reset(self, new_rank: Optional[int] = None):
        super().merge_and_reset(new_rank=new_rank)
        if new_rank is None:
            self._init_weight('weight_ab_mixer')
    
    def _del_lora(self):
        super()._del_lora()
        delattr(self, "weight_ab_mixer")

    @property
    def has_lora_weights(self):
        has_ab_mixer = hasattr(self, 'weight_ab_mixer') and self.weight_ab_mixer is not None
        return has_ab_mixer and super().has_lora_weights
    