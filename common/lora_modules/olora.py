from common.lora_modules.lora import *

class LinearWithOLoRA(LinearWithLoRA):
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
        r = self.lora_rank
        Q, R = torch.linalg.qr(weight.data)
        Qr, Rr = Q[:, :r], R[:r]
        self.weight_a.data = Rr.contiguous()
        self.weight_b.data = Qr.contiguous()
        weight.data -= self._compute_lora_weight()
        weight = weight.to(dtype)
        self.weight.data = weight