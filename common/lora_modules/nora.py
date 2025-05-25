#@author: hehaonan
#@date: 2025-04-18
"""
Implementation of NoRA(Nested Low-Rank Adaptation for Efficient Fine-Tuning Large Models), 
based on paper: https://arxiv.org/pdf/2408.10280.
"""

from torch import svd_lowrank as fast_svd
from torch.linalg import svd as standard_svd

from common.lora_modules.lora import *

class LinearWithNoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        fast_svd_n_iters: Optional[int] = 1,
    ):
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2
        super().__init__(lora_config)

    def init_lora_weights(self):
        # PiSSA share same functions with vinalla lora only with a different initialize method.
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype
        requires_grad = not self.quant

        weight = self.weight.to(torch.float32)
        if self.fast_svd:
            # Run fast svd.
            Vr, Sr, Ur = fast_svd(weight.data, self.lora_rank, niter=self.n_iters)
            Uhr = Ur.t()
            S_diag = torch.diag(Sr)
        else:
            # Full svd, which is very slow.
            V, S, Uh = standard_svd(weight.data, full_matrices=False)
            Vr, S_diag, Uhr = V[:, :self.lora_rank], torch.diag(S[:self.lora_rank]), Uh[:self.lora_rank]

        sqrt_S_diag = S_diag.sqrt_().to(dtype)
        
        self.weight_a = nn.Parameter(Uhr.to(dtype).contiguous(), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(Vr.to(dtype).contiguous(), requires_grad=requires_grad)
        self.weight_a_mixer = nn.Parameter(sqrt_S_diag[:, 1].unsqueeze(-1).contiguous(), requires_grad=requires_grad)
        self.weight_b_mixer = nn.Parameter(sqrt_S_diag[1, :].unsqueeze(0).contiguous(), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))
            self.weight_a_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_mixer_scaler = nn.Parameter(torch.Tensor(1))

        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(self._get_lora_dtype())
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(self._get_lora_dtype())
        weight_a_mixer = self._quantize_weight(self.weight_a_mixer, self.weight_a_quantizer).to(self._get_lora_dtype())
        weight_b_mixer = self._quantize_weight(self.weight_b_mixer, self.weight_b_quantizer).to(self._get_lora_dtype())
        weight_ab_mixer = torch.matmul(weight_a_mixer, weight_b_mixer)
        # weight_a = torch.matmul(weight_ab_mixer, weight_a)
        lora_result = F.linear(F.linear(F.linear(self.lora_dropout(x), weight_a), weight_ab_mixer), weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_result
    
    def _compute_lora_weight(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            weight_a_mixer = self._quantize_weight(self.weight_a_mixer, self.weight_a_quantizer).to(self._get_lora_dtype())
            weight_b_mixer = self._quantize_weight(self.weight_b_mixer, self.weight_b_quantizer).to(self._get_lora_dtype())
            weight_ab_mixer = torch.matmul(weight_a_mixer, weight_b_mixer)
            # When using vanilla lora, the ab mixer is a identical matrix

            weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            return lora_weight

    @property
    def weight_a_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_a_scaler", None)
    
    @property
    def weight_b_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_b_scaler", None)
    
    def _del_lora(self):
        super()._del_lora()
        delattr(self, "weight_a_mixer")
        delattr(self, "weight_b_mixer")

    @property
    def has_lora_weights(self):
        has_a_mixer = hasattr(self, 'weight_a_mixer') and self.weight_a_mixer is not None
        has_b_mixer = hasattr(self, 'weight_b_mixer') and self.weight_b_mixer is not None
        has_ab_mixer = has_a_mixer and has_b_mixer
        return has_ab_mixer and super().has_lora_weights
    