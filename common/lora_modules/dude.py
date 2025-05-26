from torch import svd_lowrank as fast_svd
from torch.linalg import svd as standard_svd
from common.lora_modules.lora import *

class LinearWithDude(LinearWithLoRA):
    def __init__(self,
                lora_config: LoRAConfig,
                fast_svd_n_iters: Optional[int] = 1):
        super().__init__(lora_config)
        if lora_config.lora_dropout:
            print(f'Dude is incompatible with lora dropout, skiped lora dropout')
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2

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
        else:
            # Full svd, which is very slow.
            V, S, Uh = standard_svd(weight.data, full_matrices=False)
            Vr, Sr, Uhr = V[:, :self.lora_rank], S[:self.lora_rank], Uh[:self.lora_rank]

        Sr.div_(self.lora_scaler) 
        sqrt_Sr = Sr.sqrt_()
        
        weight_a_data = torch.diag(sqrt_Sr) @ Uhr
        self.weight_a = nn.Parameter(weight_a_data.to(dtype), requires_grad=requires_grad)
        weight_b_data = Vr @ torch.diag(sqrt_Sr)
        self.weight_b = nn.Parameter(weight_b_data.to(dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)
        if not self.disable_lora:
            weight = self._apply_dora(weight)
        return F.linear(x, weight, self.bias)
    
    def _apply_dora(self, weight: torch.Tensor) -> torch.Tensor:
        # Make sure that the dtype of weight same as dtype of lora weights.
        lora_weight = self._compute_lora_weight()

        origin_weight_dtype = weight.dtype
        # Compute column-wise L2 norm.
        origin_magnitude: torch.Tensor = torch.linalg.norm(weight.detach(), dim=1).to(lora_weight.dtype)
        
        weight = weight.to(lora_weight.dtype)
        weight = weight + lora_weight
        new_magnitude: torch.Tensor = torch.linalg.norm(weight.detach(), dim=1).to(lora_weight.dtype)
        # see section 4.3 of DoRA (https://arxiv.org/abs/2402.09353)
        # "[...] we suggest treating ||V +∆V ||_c in
        # Eq. (5) as a constant, thereby detaching it from the gradient
        # graph. This means that while ||V + ∆V ||_c dynamically
        # reflects the updates of ∆V , it won’t receive any gradient
        # during backpropagation"
        new_magnitude = new_magnitude.detach()
        origin_magnitude = origin_magnitude.detach()

        # In peft. This should be added on top of the base layer output.
        # result_dora = (mag_norm_scale - 1) * (
        # F.linear(x, transpose(weight, self.fan_in_fan_out))
        # ) + mag_norm_scale * lora_result * scaling
        mag_norm_scale = (origin_magnitude / new_magnitude).view(-1, 1)
        weight = mag_norm_scale * weight
        return weight.to(origin_weight_dtype)

    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            self.weight.data = self._apply_dora(self.weight)
            return True
        return False

