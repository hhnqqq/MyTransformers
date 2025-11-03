from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *

class LinearWithRidgeLoRA(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig):
        super().__init__(lora_config)
        self.lora_alpha = lora_config.lora_scaler

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()

        r_sigma = torch.sigmoid(torch.empty(self.lora_rank, dtype=dtype).normal_(mean=2., std=1.))
        weight_ridge_data = torch.cat([
            r_sigma, 
            torch.ones(self.in_features - self.lora_rank)
            ])

        weight_a_data = torch.randn((self.in_features, self.lora_rank))
        Ua, Sa, Vha = torch.linalg.svd(weight_a_data, full_matrices=False) 

        weight_b_data = Vha.t().contiguous() / Sa @ Ua.t().contiguous() * torch.cat([
                (torch.ones(self.lora_rank) - r_sigma) * (self.lora_rank / self.lora_alpha),
                torch.zeros(self.in_features - self.lora_rank)
                ])
        self.weight_a = nn.Parameter(weight_a_data.t().contiguous().to(dtype), requires_grad=True)
        self.weight_b = nn.Parameter(weight_b_data.t().contiguous().to(dtype), requires_grad=True)
        self.weight_ridge = nn.Parameter(weight_ridge_data.contiguous().to(dtype))
        self.ridge_intensity = nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=True)
        
    def ridge_forward(self, x):
        return self.ridge_intensity.to(self._get_lora_dtype()) * (x * self.weight_ridge.to(self._get_lora_dtype()))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self.weight
        if self.disable_lora or not self.has_lora_weights:
            result = F.linear(x, weight, self.bias)
        else:
            weight_a = self.weight_a.to(self._get_lora_dtype())
            weight_b = self.weight_b.to(self._get_lora_dtype())

            lora_result = (F.linear(F.linear(self.lora_dropout(x.to(self._get_lora_dtype())), weight_a), weight_b) * self.lora_scaler + self.ridge_forward(self.lora_dropout(x))).to(x.dtype)
            result = F.linear(lora_result, weight, self.bias)
        return result

    def _compute_lora_weight(self):
        lora_weight = super()._compute_lora_weight()
        ridge_weight = (self.ridge_intensity.to(self._get_lora_dtype()) * self.weight_ridge.to(self._get_lora_dtype())).to(self.weight.dtype)
        return lora_weight + torch.diag(ridge_weight)
    
    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            lora_weight = self._compute_lora_weight()
            self.weight.data.copy_(torch.matmul(self.weight.data, lora_weight))
            # X(WBA)^T -> XA^TB^TW^T
            return True
        return False
    
    @property
    def has_lora_weights(self):
        has_ridge_intensity = hasattr(self, 'ridge_intensity') and self.ridge_intensity is not None
        has_weight_ridge = hasattr(self, "weight_ridge") and self.weight_ridge is not None
        return has_ridge_intensity and has_weight_ridge and super().has_lora_weights
    
    def _del_lora(self):
        super()._del_lora()
        delattr(self, "ridge_intensity")
        delattr(self, "weight_ridge")
