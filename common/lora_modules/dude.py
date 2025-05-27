from common.lora_modules import *

class LinearWithDude(LinearWithPiSSA, LinearWithDoRA):
    def __init__(self,
                lora_config: LoRAConfig,
                fast_svd_n_iters: Optional[int] = 1):
        LinearWithPiSSA.__init__(self, lora_config, fast_svd_n_iters)
        LinearWithDoRA.__init__(self, lora_config)

    def init_lora_weights(self):
        # PiSSA share same functions with vinalla lora only with a different initialize method.
        LinearWithPiSSA.init_lora_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the same forward function as dora.
        return LinearWithDoRA.forward(self, x)