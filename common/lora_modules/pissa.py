from torch import svd_lowrank
from common.lora_modules.lora import *

class LinearWithPiSSA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        n_iters: Optional[int] = None
    ):
        super().__init__(lora_config)
        self.n_iters = n_iters

    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else self.weight.dtype
        requires_grad = not self.quant

        weight = self.weight.to(torch.float32)
        if self.n_iters > 2:
            Vr, Sr, Ur = svd_lowrank(
                weight.data, self.lora_rank, niter=self.n_iters
            )
            Sr /= self.lora_scaler
            Uhr = Ur.t()
        else:
            V, S, Uh = torch.linalg.svd(self.weight.data, full_matrices=False)
            Vr = V[:, : self.lora_rank]
            Sr = S[: self.lora_rank]
            Sr /= self.lora_scaler
            Uhr = Uh[: self.lora_rank]
               
        self.weight_a = nn.Parameter(torch.diag(torch.sqrt(Sr)) @ Uhr, requires_grad=requires_grad, dtype=dtype)
        self.weight_b = nn.Parameter(Vr @ torch.diag(torch.sqrt(Sr)), requires_grad=requires_grad, dtype=dtype)
        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        weight = weight.data - self._compute_lora_weight()
        self.weight.data = weight.to(dtype)
