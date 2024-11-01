#@author: hehaonan
#@date: 2024-10-29
"""
Implementation of PiSSA(PRINCIPAL SINGULAR VALUES AND SINGULAR VECTORS ADAPTATION), 
a initialize method for lora, based on paper: https://arxiv.org/pdf/2404.02948.

In PiSSA:
Origin weight w is decomposied by SVD (W = USV^T)
Where U, V are the singular vectors with orthonormal columns
A=U[:,:r]S1/2[:r,:r] ∈ Rmxr, B =S1/2[:r,:r]V^T[:,:r] ∈ Rrxn
Residual W_{res} = W - AB
"""

from torch import svd_lowrank as fast_svd
from torch.linalg import svd as standard_svd

from common.lora_modules.lora import *

class LinearWithPiSSA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        fast_svd_n_iters: Optional[int] = 1,
    ):
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2
        super().__init__(lora_config)

    def _init_lora_weights(self):
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
            V, S, Uh = standard_svd(self.weight.data, full_matrices=False)
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
