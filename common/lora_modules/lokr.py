import math
from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *
from torch import Tensor

class LinearWithLoKr(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig, k: int = 32, weight_c_init_method = None):
        super().__init__(lora_config)
        self.in_d = max(u for u in range(1, min(k, int(math.sqrt(self.in_features))) + 1) if self.in_features % u == 0)
        self.out_d = max(u for u in range(1, min(k, int(math.sqrt(self.out_features))) + 1) if self.out_features % u == 0)
        self.weight_c_init_method = weight_c_init_method

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_d), dtype=dtype), requires_grad=True)
        self.weight_b = nn.Parameter(torch.zeros((self.out_d, self.lora_rank), dtype=dtype), requires_grad=True)
        self.weight_c = nn.Parameter(torch.empty((self.out_features // self.out_d, self.in_features // self.in_d), dtype=dtype), requires_grad=True)

        self._init_weight('weight_a')
        self._init_weight('weight_b')
        self._init_weight('weight_c')

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        if weight_name == 'weight_c':
            weight_name = 'weight_a'
        return super().get_weight_init_kwargs(weight_name, method)
    
    def _compute_lora_weight(self):
        dtype = self._get_lora_dtype()
        lora_weight = self.lora_scaler * (torch.kron(torch.matmul(self.weight_b.to(dtype), self.weight_a.to(dtype)), self.weight_c.to(dtype)))
        return lora_weight.to(self.weight.dtype)
    
    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        lora_weight = self._compute_lora_weight()
        lora_result = torch.matmul(x, lora_weight).to(self.weight.dtype)
        return result + lora_result
    
    @property
    def has_lora_weights(self):
        has_weight_c = hasattr(self, 'weight_c') and self.weight_c is not None
        return has_weight_c and super().has_lora_weights
    
    def _del_lora(self):
        super()._del_lora()
        delattr(self, "weight_c")