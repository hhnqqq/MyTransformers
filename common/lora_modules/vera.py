# @author: haonan he
# @date: 2024-10-28
"""Implementation of VeRA (Vector-based Random Matrix Adaptation) based on the paper:
https://arxiv.org/abs/2310.11454.

VeRA builds upon LoRA, where LoRA updates the weight matrix W by training two
low-rank matrices, A and B, with an intermediate rank r. In VeRA, these matrices
are frozen, shared across all layers, and adapted using trainable vectors d and b.
This significantly reduces the number of trainable parameters.

In both approaches, the low-rank matrices and vectors can be merged into the
original weight matrix W without introducing additional inference latency. This
means VeRA can maintain a much smaller number of trainable parameters even with
a larger rank.

Note that this implementation does not share the same weights b and a across the
model. For the `shared across layers` implementation, please refer to:
https://github.com/huggingface/peft/blob/main/src/peft/tuners/vera/model.py#L43

For the ablation study on shared parameters, we quote from the paper:
`Sharing Random Matrices: We conduct experiments on RTE, MRPC, CoLA, and STS-B
tasks to assess the impact of sharing random matrices on performance. We evaluate
two setups: one with random matrices shared across all adapted layers, and another
with uniquely generated matrices for each layer. Results in Table 7 show that the
mean performance is identical for RTE and STS-B tasks, and there is a slight
improvement for MRPC and CoLA when using unique matrices.`"""

import torch

from common.lora_modules.lora import *
    
class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        lora_config: LoRAConfig,
        lambda_b_init_method: str = 'zero',
        lambda_d_init_method: str = 'small_constant'
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            lambda_b_init_method (str, optional): Initialization method for lambda b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zeros'.
            lambda_d_init_method (str, optional): Initialization method for lambda d. ['zero', 'ones', 'small_constant', 'random']. Default is 'small_constant'.
        """
        super().__init__(lora_config)
        self._init_lambdas(lambda_b_init_method, lambda_d_init_method)
        if lora_config.weight_b_init_method is None:
            raise ValueError('The init method for weight b in vera can not be zero.')
    
    def _init_lambdas(self, b_init_method: str, d_init_method: str):
        """
        Initialize lambdas with different methods.
        Args:
            b_init_method: The method to initialize lambda b.
            d_init_method: The method to initialize lambda d.
        """
        # Initialize vector b
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant
        if b_init_method == 'zero':
            lambda_b = torch.zeros(self.out_features, dtype=dtype)
        elif b_init_method == 'ones':
            lambda_b = torch.ones(self.out_features, dtype=dtype)
        elif b_init_method == 'small_constant':
            lambda_b = 0.1 * torch.ones(self.out_features, dtype=dtype)
        elif b_init_method == 'random':
            lambda_b = torch.rand(self.out_features, dtype=dtype)
        else:
            raise ValueError(f"Unknown b_init_method: {b_init_method}")
        

        # Initialize vector d
        if d_init_method == 'zero':
            lambda_d = torch.zeros(self.lora_rank, dtype=dtype)
        elif d_init_method == 'ones':
            lambda_d = torch.ones(self.lora_rank, dtype=dtype)
        elif d_init_method == 'small_constant':
            lambda_d = 0.1 * torch.ones(self.lora_rank, dtype=dtype)
        elif d_init_method == 'random':
            lambda_d = torch.rand(self.lora_rank, dtype=dtype)
        else:
            raise ValueError(f"Unknown d_init_method: {d_init_method}")
        self.lambda_b = nn.Parameter(lambda_b, requires_grad=requires_grad)
        self.lambda_d = nn.Parameter(lambda_d, requires_grad=requires_grad)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # Called by forward method in LinearWithLoRA
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(self._get_lora_dtype())
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(self._get_lora_dtype())
        lambda_b = self.lambda_b.to(self._get_lora_dtype())
        lambda_d = self.lambda_d.to(self._get_lora_dtype())
        lora_result = lambda_b * F.linear(
            lambda_d * F.linear(
                self.lora_dropout(x), 
                weight_a
            ), 
            weight_b
        )

        return result + self.lora_scaler * lora_result.to(result.dtype)
    
    def _compute_lora(self):
        # Called by merge lora method in LinearWithLoRA
        if self.has_lora_weights:
            # Compute adapted lora weights.
            weight_a = (
                self._quantize_weight(self.weight_a, self.weight_a_quantizer)
                .to(self._get_lora_dtype()) * self.lambda_d.to(self._get_lora_dtype())
            )
            weight_b = (
                self._quantize_weight(self.weight_b, self.weight_b_quantizer)
                .to(self._get_lora_dtype()) * self.lambda_b.to(self._get_lora_dtype())
            )

            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)

            return lora_weight.to(self.weight.dtype)