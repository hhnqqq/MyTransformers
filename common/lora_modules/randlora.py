"""
This is a naive implementation for RandLoRA, optimized version will be uploaded soon!
"""
import math
from common.lora_modules.lora import *
    
class UniqueBaseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight_a, randlora_lambda, randlora_gemma):
        out = randlora_lambda[:, :, None] * weight_a * randlora_gemma[None, : , :]
        ctx.save_for_backward(weight_a, randlora_lambda, randlora_gemma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        weight_a, randlora_lambda, randlora_gamma = ctx.saved_tensors
        weight_a, randlora_lambda, randlora_gamma = (
            weight_a.to(grad_output.dtype),
            randlora_lambda.to(grad_output.dtype),
            randlora_gamma.to(grad_output.dtype),
        )
        grad_randlora_lambda = torch.einsum("kbj,kvj,bj->kb", grad_output, weight_a, randlora_gamma)
        grad_randlora_gamma = torch.einsum("kbj,kvj,kb->bj", grad_output, weight_a, randlora_lambda)
        return None, grad_randlora_lambda, grad_randlora_gamma


class LinearWithRandLoRA(LinearWithLoRA):
    def __init__(self, 
                lora_config: LoRAConfig):
        super().__init__(lora_config)
        
        self.min_features = min(self.in_features, self.out_features)

        if lora_config.weight_b_init_method is None:
            raise ValueError('The init method for weight b in randlora can not be zero.')
        if lora_config.quant:
            print(f'Currently RandLoRA is incompatible with quant, skipped quant')
            
        self.num_loras = math.ceil(self.min_features / self.lora_rank)

    def init_lora_weights(self):
        dtype = self._get_lora_dtype()
        requires_grad = True

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, 1, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.num_loras, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        
        self.randlora_gemma = nn.Parameter(
            torch.ones(self.num_loras, self.in_features, dtype=dtype)
            / self.out_features,
            requires_grad=True
        )
        
        self.randlora_lambda = nn.Parameter(torch.randn(self.lora_rank, self.num_loras, dtype=dtype), requires_grad=True)

        self._init_weight('weight_a')
        self._init_weight('weight_b')
        # see https://github.com/PaulAlbert31/RandLoRA/blob/main/peft/src/peft/tuners/randlora/model.py line 193
        self.weight_a.data = self.weight_a.data / self.weight_a.data.std()
        self.weight_b.data = self.weight_b.data / self.weight_b.data.std()
            

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # [r, 1, in] -> [r, n, in]
        weight_a = self.weight_a.to(self._get_lora_dtype()).repeat(1, self.num_loras, 1)
        # [r, n]
        randlora_lambda = self.randlora_lambda.to(self._get_lora_dtype())
        # [n, in]
        randlora_gemma = self.randlora_gemma.to(self._get_lora_dtype())
        # [r, n, in] -> [r*n, in]
        weight_a = UniqueBaseGrad.apply(weight_a, randlora_lambda, randlora_gemma).flatten(end_dim=1)
        # [out, n, r] -> [out, n*r]
        weight_b = self.weight_b.to(self._get_lora_dtype()).flatten(start_dim=1)
        
        # [bsz, seq_len, in][r*n, in]^T -> [bsz, seq_len, r*n]d[out, n*r]^T -> [bsz, seq_len, out]
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b)


        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora(self):
        if self.has_lora_weights:
            weight_a = self.weight_a.to(self._get_lora_dtype()).repeat(1, self.num_loras, 1)
            weight_b = self.weight_b.to(self._get_lora_dtype()).flatten(start_dim=1)
            randlora_lambda = self.randlora_lambda.to(self._get_lora_dtype())
            randlora_gemma = self.randlora_gemma.to(self._get_lora_dtype())
            weight_a = randlora_lambda[:, :, None] * weight_a * randlora_gemma[None, : , :]
            
            lora_weight = torch.zeros(self.out_features, self.in_features, device=weight_a.device, dtype=weight_a.dtype)
            
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)
        
    @property
    def has_lora_weights(self):
        has_lambda = hasattr(self, 'randlora_lambda') and self.randlora_lambda is not None
        has_gemma = hasattr(self, 'randlora_gemma') and self.randlora_gemma is not None
        has_lambda_gemma = has_lambda and has_gemma
        return has_lambda_gemma and super().has_lora_weights