import math
from common.lora_modules.lora import *
    
class UniqueBaseGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, shared_weight_a, randlora_lambda, randlora_gemma):
        out = randlora_lambda[:, :, None] * shared_weight_a * randlora_gemma[None, : , :]
        ctx.save_for_backward(shared_weight_a, randlora_lambda, randlora_gemma)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        shared_weight_a, randlora_lambda, randlora_gamma = ctx.saved_tensors
        shared_weight_a, randlora_lambda, randlora_gamma = (
            shared_weight_a.to(grad_output.dtype),
            randlora_lambda.to(grad_output.dtype),
            randlora_gamma.to(grad_output.dtype),
        )
        grad_randlora_lambda = torch.einsum("kbj,kvj,bj->kb", grad_output, shared_weight_a, randlora_gamma)
        grad_randlora_gamma = torch.einsum("kbj,kvj,kb->bj", grad_output, shared_weight_a, randlora_lambda)
        return None, grad_randlora_lambda, grad_randlora_gamma


class LinearWithRandLoRA(LinearWithLoRA):
    def __init__(self, 
                lora_config: LoRAConfig):
        super().__init__(lora_config)
        self.share_lora_weights = True
        if lora_config.quant:
            print(f'Currently RandLoRA is incompatible with quant, skipped quant')

    def update_shared_weights(
        self,
        shared_weight_a,
        shared_weight_b
    ):
        """
        Update the low-rank weights of this layer using shared weights.
        And initialize trainable verctors according the min dimension of this layer.
        The number of trainable parameters of this layer is:
            math.ceil(hidden_dim / lora_rank) * (self.min_dim + lora_rank)
        """
        dtype = self._get_lora_dtype()
        self.shared_weight_a = shared_weight_a
        self.shared_weight_b = shared_weight_b
        self.num_loras = shared_weight_b.shape[1]
        self.min_dim = min(self.out_features, self.in_features)
        self.max_dim = max(self.out_features, self.in_features)
        self.randlora_gemma = nn.Parameter(
            torch.ones(self.num_loras, self.min_dim, dtype=dtype)
            / self.max_dim,
            requires_grad=True
        )
        
        self.randlora_lambda = nn.Parameter(torch.randn(self.lora_rank, self.num_loras, dtype=dtype), requires_grad=True)

    def init_lora_weights(self):
        pass
            
    
    def _get_sliced_lora_weights(self):
        shared_weight_a = self.shared_weight_a[:self.lora_rank, :, :self.min_dim]
        shared_weight_b = self.shared_weight_b[:self.max_dim, :, :self.lora_rank]

        # [r, 1, min] -> [r, n, min]
        shared_weight_a = shared_weight_a.to(self._get_lora_dtype()).repeat(1, self.num_loras, 1)
        # [r, n]
        randlora_lambda = self.randlora_lambda.to(self._get_lora_dtype())
        # [n, min]
        randlora_gemma = self.randlora_gemma.to(self._get_lora_dtype())
        # [r, n, min] -> [r*n, min]
        shared_weight_a = UniqueBaseGrad.apply(shared_weight_a, randlora_lambda, randlora_gemma).flatten(end_dim=1)
        # [max, n, r] -> [max, n*r]
        shared_weight_b = shared_weight_b.to(self._get_lora_dtype()).flatten(start_dim=1)
        if self.min_dim == self.out_features:
            # Make sure that out_features corresponding to shared_weight_b.
            return shared_weight_a, shared_weight_b
        else:
            return shared_weight_b.T, shared_weight_a.T
        
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        shared_weight_a, shared_weight_b = self._get_sliced_lora_weights()
        
        # [bsz, seq_len, in][r*n, in]^T -> [bsz, seq_len, r*n]d[out, n*r]^T -> [bsz, seq_len, out]
        lora_result = F.linear(F.linear(self.lora_dropout(x), shared_weight_a), shared_weight_b)


        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora(self):
        if self.has_lora_weights:
            shared_weight_a = self.shared_weight_a[:self.lora_rank, :, :self.min_dim]
            shared_weight_b = self.shared_weight_b[:self.max_dim, :, :self.lora_rank]

            shared_weight_a = shared_weight_a.to(self._get_lora_dtype()).repeat(1, self.num_loras, 1)
            shared_weight_b = shared_weight_b.to(self._get_lora_dtype()).flatten(start_dim=1)
            
            randlora_lambda = self.randlora_lambda.to(self._get_lora_dtype())
            randlora_gemma = self.randlora_gemma.to(self._get_lora_dtype())
            shared_weight_a = randlora_lambda[:, :, None] * shared_weight_a * randlora_gemma[None, : , :]
            
            lora_weight = torch.zeros(self.out_features, self.in_features, device=shared_weight_a.device, dtype=shared_weight_a.dtype)
            
            lora_weight = self.lora_scaler * torch.matmul(shared_weight_b, shared_weight_a)
            if self.max_dim == self.out_features:
                return lora_weight.to(self.weight.dtype)
            else:
                return lora_weight.T.to(self.weight.dtype)
        
    @property
    def has_lora_weights(self):
        has_lambda = hasattr(self, 'randlora_lambda') and self.randlora_lambda is not None
        has_gemma = hasattr(self, 'randlora_gemma') and self.randlora_gemma is not None
        has_lambda_gemma = has_lambda and has_gemma
        return has_lambda_gemma and super().has_lora_weights
    
def prepare_shared_lora_weights_randlora(model: nn.Module, args) -> tuple[nn.Parameter, nn.Parameter]:
    """
    Prepare shared LoRA weights that will be used across all layers.
    
    Args:
        model: The model containing LoRA layers
        args: Arguments containing LoRA configuration
        
    Returns:
        Tuple of (shared_weight_a, shared_weight_b) Parameters
    """
    # Find the maximum dimensions needed across all layers
    max_in_features = 0
    max_out_features = 0
    
    for module in model.modules():
        if isinstance(module, LinearWithRandLoRA):
            max_in_features = max(max_in_features, module.in_features)
            max_out_features = max(max_out_features, module.out_features)

    num_loras = math.ceil(min(max_in_features, max_out_features) / args.lora_rank)

    if max_in_features == 0 or max_out_features == 0:
        raise ValueError("No LinearWithRandLoRA layers found in the model")
    
    # Create shared parameters
    dtype = torch.float32 if args.run_lora_in_fp32 else torch.float16
    device = next(model.parameters()).device
    
    # Initialize A matrix
    shared_weight_a = nn.Parameter(
        torch.empty((args.lora_rank, 1, min(max_in_features, max_out_features)), dtype=dtype, device=device))
    
    # Initialize B matrix  
    shared_weight_b = nn.Parameter(
        torch.empty((max(max_in_features, max_out_features), num_loras, args.lora_rank), dtype=dtype, device=device))
    
    # Initialize weights
    with torch.no_grad():
        if args.weight_a_init_method == 'kaiming':
            nn.init.kaiming_uniform_(shared_weight_a, a=5**0.5, mode='fan_in')
        else:
            nn.init.normal_(shared_weight_a, mean=0.0, std=1 / (max_in_features ** 0.5))
        
        if args.weight_b_init_method == 'kaiming':
            nn.init.kaiming_uniform_(shared_weight_b, a=5**0.5, mode='fan_in')
        else:
            nn.init.normal_(shared_weight_b, mean=0.0, std=0.02)

    # see https://github.com/PaulAlbert31/RandLoRA/blob/main/peft/src/peft/tuners/randlora/model.py line 193
    shared_weight_a.data = shared_weight_a.data / shared_weight_a.data.std()
    shared_weight_b.data = shared_weight_b.data / shared_weight_a.data.std()

    return shared_weight_a, shared_weight_b