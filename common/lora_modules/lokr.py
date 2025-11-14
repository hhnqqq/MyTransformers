# @author：hehaonan
# @date： 2025/11/14
"""
An unofficial implementation of Lokr (https://arxiv.org/abs/2309.14859)
"""
import math
from common.lora_modules.lora import LoRAConfig
from common.lora_modules.lora import *

class LinearWithLoKr(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, k: int = 32, decompose_weight_c = False, weight_c_init_method = None):
        super().__init__(lora_config)
        # The in_features of weight_a is the largest interger smaller than k and can devide in_features of pre-trained weight evenly.
        self.in_d = max(u for u in range(1, min(k, int(math.sqrt(self.in_features))) + 1) if self.in_features % u == 0)
        self.out_d = max(u for u in range(1, min(k, int(math.sqrt(self.out_features))) + 1) if self.out_features % u == 0)
        # weight_c can be decomposed further to reduce the total trainable parameter count.
        self.decompose_weight_c = decompose_weight_c
        self.weight_c_init_method = weight_c_init_method
        self.weight_ca_init_method = weight_c_init_method
        self.weight_cb_init_method = weight_c_init_method

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_d), dtype=dtype), requires_grad=True)
        self.weight_b = nn.Parameter(torch.zeros((self.out_d, self.lora_rank), dtype=dtype), requires_grad=True)
        self._init_weight('weight_a')
        self._init_weight('weight_b')

        weight_c_in, weight_c_out = self.in_features // self.in_d, self.out_features // self.out_d
        # Make sure the number of trainable parameter after decomposition smaller than the the original weight_c.
        if self.decompose_weight_c and (self.lora_rank * (weight_c_out + weight_c_in)) < weight_c_out * weight_c_in:
            self.weight_ca = nn.Parameter(torch.empty((self.lora_rank, weight_c_in), dtype=dtype), requires_grad=True)
            self.weight_cb = nn.Parameter(torch.empty((weight_c_out, self.lora_rank), dtype=dtype), requires_grad=True)
            self._init_weight('weight_ca')
            self._init_weight('weight_cb')
        else:
            self.weight_c = nn.Parameter(torch.empty((weight_c_out, weight_c_in), dtype=dtype), requires_grad=True)
            self._init_weight('weight_c')

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        if 'weight_c' in weight_name:
            weight_name = 'weight_a'
        return super().get_weight_init_kwargs(weight_name, method)
    
    def _compute_lora_weight(self):
        dtype = self._get_lora_dtype()
        weight_c = torch.matmul(self.weight_cb.to(dtype), self.weight_ca.to(dtype)) if self.decompose_weight_c else self.weight_c.to(dtype)
        lora_weight = self.lora_scaler * (torch.kron(torch.matmul(self.weight_b.to(dtype), self.weight_a.to(dtype)), weight_c))
        return lora_weight.to(self.weight.dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self.weight
        if not self.disable_lora:
            weight = weight + self._compute_lora_weight()
        return F.linear(x, weight, self.bias)
    
    @property
    def has_lora_weights(self):
        if self.decompose_weight_c:
            has_weight_ca = hasattr(self, 'weight_ca') and self.weight_ca is not None
            has_weight_cb = hasattr(self, 'weight_cb') and self.weight_cb is not None
            has_weight_c = has_weight_ca and has_weight_cb
        else:
            has_weight_c = hasattr(self, 'weight_c') and self.weight_c is not None
        return has_weight_c and super().has_lora_weights
    
    def _del_lora(self):
        super()._del_lora()
        if self.decompose_weight_c:
            delattr(self, "weight_ca")
            delattr(self, "weight_cb")
        else:
            delattr(self, "weight_c")