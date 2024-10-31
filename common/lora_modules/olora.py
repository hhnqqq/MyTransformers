from common.lora_modules.lora import *

class LinearWithOLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
    ):
        super().__init__(lora_config)

    def _init_lora_weights(self):
        dtype = self._get_lora_dtype()
        weight_dtype = self.weight.dtype
        requires_grad = not self.quant

        weight = self.weight.to(torch.float32)
        r = self.lora_rank
        Q, R = torch.linalg.qr(weight.data)
        Qr, Rr = Q[:, :r], R[:r]
        self.weight_a = nn.Parameter(Rr.contiguous().to(dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(Qr.contiguous().to(dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))
            
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)