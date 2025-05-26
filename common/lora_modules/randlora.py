"""
This is a naive implementation for RandLoRA, optimized version will be uploaded soon!
"""
import math
from common.lora_modules.lora import *
    
class LinearWithRandLoRA(LinearWithLoRA):
    def __init__(self, 
                lora_config: LoRAConfig,
                lambda_b_init_method: str = 'zero',
                lambda_d_init_method: str = 'small_constant'):
        super().__init__(lora_config)
        self.lambda_b_init_method = lambda_b_init_method
        self.lambda_d_init_method = lambda_d_init_method
        
        self.min_features = min(self.in_features, self.out_features)

        if lora_config.weight_b_init_method is None:
            raise ValueError('The init method for weight b in randlora can not be zero.')
        if lora_config.quant:
            print(f'Currently RandLoRA is incompatible with quant, skipped quant')
            
        self.num_loras = math.ceil(self.min_features / self.lora_rank)
        self.init_methods = {
            'zero': lambda size: torch.zeros(size),
            'ones': lambda size: torch.ones(size),
            'small_constant': lambda size: 0.1 * torch.ones(size),
            'random': lambda size: torch.rand(size)
        }

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        requires_grad = True

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.num_loras, self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        
        self._init_weight('weight_a')
        self._init_weight('weight_b')
        self._init_lambdas()

    def _init_weight(self, weight_name: str):
        weight = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        
        if weight_name == 'weight_b':
            for i in range(self.num_loras):
                self.get_weight_init_method(**init_kwargs)(weight[i])
        else:
            self.get_weight_init_method(**init_kwargs)(weight)
            
    def _init_lambdas(self):
        dtype = self._get_lora_dtype()
        requires_grad = True

        if self.lambda_b_init_method not in self.init_methods:
            raise ValueError(f"Unknown initialization method: {self.lambda_b_init_method}")
        if self.lambda_d_init_method not in self.init_methods:
            raise ValueError(f"Unknown initialization method: {self.lambda_d_init_method}")
            
        self.lambda_b = nn.Parameter(
            torch.stack([self.init_methods[self.lambda_b_init_method](self.out_features) 
                        for _ in range(self.num_loras)]).to(dtype), 
            requires_grad=requires_grad
        )
        
        self.lambda_d = nn.Parameter(
            torch.stack([self.init_methods[self.lambda_d_init_method](self.lora_rank) 
                        for _ in range(self.num_loras)]).to(dtype), 
            requires_grad=requires_grad
        )

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        lambda_b = self.lambda_b.to(self._get_lora_dtype())
        lambda_d = self.lambda_d.to(self._get_lora_dtype())
        
        x_dropped = self.lora_dropout(x)
        intermediate = F.linear(x_dropped, weight_a)
        
        lora_result = torch.zeros(*x.shape[:2], self.out_features, device=x.device, dtype=result.dtype)
        
        for i in range(self.num_loras):
            scaled_intermediate = intermediate * lambda_d[i].unsqueeze(0).unsqueeze(0)
            lora_contribution = F.linear(scaled_intermediate, weight_b[i]) * lambda_b[i].unsqueeze(0).unsqueeze(0)
            lora_result += lora_contribution

        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora(self):
        weight_a = self.weight_a.to(self._get_lora_dtype())
        weight_b = self.weight_b.to(self._get_lora_dtype())
        lambda_b = self.lambda_b.to(self._get_lora_dtype())
        lambda_d = self.lambda_d.to(self._get_lora_dtype())
        
        lora_weight = torch.zeros(self.out_features, self.in_features, device=weight_a.device, dtype=weight_a.dtype)
        
        if self.has_lora_weights:
            for i in range(self.num_loras):
                scaled_weight_a = weight_a * lambda_d[i].unsqueeze(1)
                scaled_weight_b = weight_b[i] * lambda_b[i].unsqueeze(1)
                
                lora_weight += self.lora_scaler * torch.matmul(scaled_weight_b, scaled_weight_a)

        return lora_weight.to(self.weight.dtype)
        
    @property
    def has_lora_weights(self):
        has_lambda_b = hasattr(self, 'lambda_b') and self.lambda_b is not None
        has_lambda_d = hasattr(self, 'lambda_d') and self.lambda_d is not None
        has_lambdas = has_lambda_b and has_lambda_d
        return has_lambdas and super().has_lora_weights