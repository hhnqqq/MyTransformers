from common.lora_modules.lora import LoRAConfig
from common.lora_modules.qlora import *
from torch import Tensor

class LinearWithLoHA(LinearWithQLoRA):
    def __init__(self, lora_config: LoRAConfig):
        super().__init__(lora_config)
        if lora_config.lora_dropout:
            self.lora_dropout = nn.ModuleList([nn.Dropout(lora_config.lora_dropout) for _ in range(2)])
        else:
            self.lora_dropout = nn.ModuleList([nn.Identity() for _ in range(2)])

    def init_lora_weights(self):
        dtype = None
        requires_grad = True

        self.weight_a = nn.ParameterList([nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad) for _ in range(2)])
        self.weight_b = nn.ParameterList([nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad) for _ in range(2)])

        self._init_weight('weight_a')
        self._init_weight('weight_b')

    def _init_weight(self, weight_name: str):
        weight_list = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        for weight in weight_list:
            self.get_weight_init_method(**init_kwargs)(weight)

    def _compute_lora_weight(self):
        lora_dtype = self._get_lora_dtype()
        lora_weight_1 = torch.matmul(self.weight_b[0].to(lora_dtype), self.weight_a[0].to(lora_dtype))
        lora_weight_2 = torch.matmul(self.weight_b[1].to(lora_dtype), self.weight_a[1].to(lora_dtype))

        lora_weight = self.lora_scaler * lora_weight_1 * lora_weight_2
        return lora_weight.to(self.weight.dtype)
    
    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        lora_dtype = self._get_lora_dtype()
        lora_result_1 = F.linear(F.linear(self.lora_dropout[0](x), self.weight_a[0].to(lora_dtype)), self.weight_b[0].to(lora_dtype))
        lora_result_2 = F.linear(F.linear(self.lora_dropout[1](x), self.weight_a[1].to(lora_dtype)), self.weight_b[1].to(lora_dtype))
        
        return result + (self.lora_scaler * lora_result_1 * lora_result_2)