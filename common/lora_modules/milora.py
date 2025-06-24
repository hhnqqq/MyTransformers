from torch.linalg import svd as standard_svd
from torch import svd_lowrank as fast_svd
from common.lora_modules.lora import *

def to_torch_tensor(np_array):
    if np_array.strides[0] < 0:
        np_array = np_array.copy()
    return torch.from_numpy(np_array)

class LinearWithMILoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        fast_svd_n_iters: Optional[int] = 1,
    ):
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2
        super().__init__(lora_config)

    def init_lora_weights(self):
        # MILoRA share same functions with vinalla lora only with a different initialize method.
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype
        requires_grad = not self.quant

        weight = self.weight.to(torch.float32)
        if self.fast_svd:
            # Run fast svd.
            Vr, Sr, Ur = fast_svd(weight.data, self.lora_rank, niter=self.n_iters)
            Vhr = Vr.t()
        else:
            # Full svd, which is very slow.
            U, S, Vh = standard_svd(weight.data, full_matrices=False)
            Ur, Sr, Vhr = U[:, -self.lora_rank:], S[-self.lora_rank:], Vh[-self.lora_rank:]

        Sr.div_(self.lora_scaler) 
        sqrt_Sr = Sr.sqrt_()
        
        weight_a_data = torch.diag(sqrt_Sr) @ Vhr
        self.weight_a = nn.Parameter(weight_a_data.to(dtype), requires_grad=requires_grad)
        weight_b_data = Ur @ torch.diag(sqrt_Sr)
        self.weight_b = nn.Parameter(weight_b_data.to(dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)
