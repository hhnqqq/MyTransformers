from torch import svd_lowrank
from common.lora_modules.lora import *

class LinearWithPiSSA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        n_iters: Optional[int] = None,
        fast_svd: bool = True
    ):
        self.n_iters = n_iters
        self.fast_svd = fast_svd
        super().__init__(lora_config)

    def _init_lora_weights(self):
        # PiSSA share same functions with vinalla lora only with a different initialize method.
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype
        requires_grad = not self.quant

        weight = self.weight.to(torch.float32)
        if self.n_iters > 2:
            # Run fast svd.
            Vr, Sr, Ur = svd_lowrank(weight.data, self.lora_rank, niter=self.n_iters)
            Uhr = Ur.t()
        else:
            # Full svd, which is very slow.
            V, S, Uh = torch.linalg.svd(self.weight.data, full_matrices=False)
            Vr, Sr, Uhr = V[:, :self.lora_rank], S[:self.lora_rank], Uh[:self.lora_rank]
        Sr /= self.lora_scaler
               
        sqrt_Sr = torch.sqrt(Sr)
        self.weight_a = nn.Parameter((torch.diag(sqrt_Sr) @ Uhr).to(dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter((Vr @ torch.diag(sqrt_Sr)).to(dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)
