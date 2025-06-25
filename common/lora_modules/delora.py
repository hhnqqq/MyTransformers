from common.lora_modules.lora import *
from common.lora_modules.lora import LoRAConfig
from torch import Tensor

class LinearWithDELoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, delora_lambda):
        super().__init__(lora_config)
        self.delora_lambda = delora_lambda

    def init_lora_weights(self):
        super().init_lora_weights()
        self.delora_lambda = nn.Parameter(torch.full((1,), self.delora_lambda))
        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)
        self.weight.data = self.weight.data - self._compute_lora_weight()
        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)

    def _compute_lora_weight(self):
        # Get weights
        dtype = self._get_lora_dtype()
        device = self.weight.device
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(dtype).to(device)
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(dtype).to(device)
        delora_lambda = self.delora_lambda.to(dtype).to(device)

        # Get norms
        weight_a_norm = weight_a.norm(dim=1)
        weight_b_norm = weight_b.norm(dim=0)

        # AB normalization
        diag = torch.div(delora_lambda / self.lora_rank, torch.mul(weight_a_norm, weight_b_norm))
        diag = torch.diag_embed(diag)

        # Get ABCD
        lora_weight = weight_b @ diag @ weight_a

        self.Wnorm = self.weight.data.norm(dim=0).unsqueeze(0)
        lora_weight = torch.mul(lora_weight, self.Wnorm)

        return lora_weight
    
    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        lora_result = F.linear(self.lora_dropout(x), self._compute_lora_weight())
        return result + lora_result
    
    @property
    def has_lora_weights(self):
        """
        Check if this layer has DELoRA weights.
        """
        has_init_a = hasattr(self, 'frozen_a') and self.frozen_a is not None
        has_init_b = hasattr(self, 'frozen_b') and self.frozen_b is not None
        return has_init_a and has_init_b and super().has_lora_weights

    def _del_lora(self):
        super()._del_lora()
        if hasattr(self, 'frozen_a'):
            delattr(self, "frozen_a")
        if hasattr(self, 'frozen_b'):
            delattr(self, "frozen_b")
