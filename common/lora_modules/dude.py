from common.lora_modules.dora import *
from common.lora_modules.pissa import LinearWithPiSSA

class LinearWithDude(LinearWithPiSSA, LinearWithDoRA):
    def __init__(self,
                lora_config: LoRAConfig,
                fast_svd_n_iters: Optional[int] = 1):
        LinearWithPiSSA.__init__(self, lora_config, fast_svd_n_iters)

    def init_lora_weights(self):
        # PiSSA share same functions with vinalla lora only with a different initialize method.
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant
        self.origin_magnitude = nn.Parameter(
            torch.linalg.norm(self.weight.detach(), dim=1).to(dtype=dtype),
            requires_grad=requires_grad
        )
        LinearWithPiSSA.init_lora_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the same forward function as dora.
        return LinearWithDoRA.forward(self, x)