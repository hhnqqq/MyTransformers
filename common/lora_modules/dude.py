# @author: haonan he
"""
Implementaion of DuDe(Dual Decomposition of Weights and Singular Value Low Rank Adaptation) [arxiv preprint]
Paper link: https://arxiv.org/abs/2505.14367
Code reference: None

DuDe decomposes weight matrices into magnitude and direction components, 
employing Singular Value Decomposition (SVD) for principled initialization.
"""
from common.lora_modules.dora import *
from common.lora_modules.pissa import LinearWithPiSSA

class LinearWithDude(LinearWithPiSSA, LinearWithDoRA):
    def __init__(self,
                lora_config: LoRAConfig,
                fast_svd_n_iters: Optional[int] = 1):
        LinearWithPiSSA.__init__(self, lora_config, fast_svd_n_iters)

    def init_lora_weights(self):
        # PiSSA share same functions with vinalla lora only with a different initialize method.
        LinearWithDoRA.init_origin_magnitude(self)
        LinearWithPiSSA.init_lora_weights(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use the same forward function as dora.
        return LinearWithDoRA.forward(self, x)