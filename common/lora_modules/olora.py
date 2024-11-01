#@author: hehaonan
#@date: 2024-10-31
"""
Implementation of OLoRA(Orthonormal Low-Rank Adaptation), a initialize method for lora, based on paper:
https://arxiv.org/pdf/2406.01775.

Orthonormality in NN has several benefits:
1. Orthonormal matrices help maintain the norm of gradients during backpropagation.
2. Mitigating issues like vanishing or exploding gradients that can hinder convergence.
3. The orthogonal group, to which orthonormal matrices belong, exhibits favorable geometric 
   properties that can translate to a better conditioned optimization landscape.

In OLoRA:
Origin weight w is decomposied to QR (Q \in R^{mxr}, R \in R^{rxn})
where Q is an orthogonal matrix , R is an upper triangular matrix
B = Q[:, :r], A =R[:r,:]
Residual W_{res} = W - AB
"""

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