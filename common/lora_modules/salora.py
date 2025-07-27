"""
Un-offcial implementation of SALoRA(Structure aware LoRA).
SALoRA introduces a leanable gate to dis-activate unimportant ranks.
The value of the gate is optimized by a L1 norm loss
Similar to AdaLoRA, the features of A and B are constrained by a orthogonality loss.
"""
import math
import torch.nn.functional as F

from common.lora_modules.lora import *

class HardConcreteGate(nn.Module):
    """
    Implements Hard-Concrete (HC) distribution for learnable sparsity
    Based on "Structured Pruning of Neural Networks with Budget-Aware Regularization"
    and "Learning Sparse Neural Networks through L0 Regularization"
    """
    # Sugeested hyper-parameters reported in the paper.
    def __init__(
        self,
        size,          # Size of the gate tensor
        init_value=1.0, # Initial log-alpha value
        beta=1.0,       # Temperature parameter
        gamma=-0.1,     # Stretch parameter left endpoint 
        zeta=1.1,       # Stretch parameter right endpoint
    ):
        """
        Initialize the Hard-Concrete gate module
        
        Args:
            size: Size/shape of the gate tensor
            init_value: Initial value for log_alpha
            loc_mean: Mean value for initializing log_alpha
            loc_sdev: Std dev for initializing log_alpha
            beta: Temperature parameter controlling the discreteness
            gamma: Stretch parameter (left endpoint)
            zeta: Stretch parameter (right endpoint)
            fix_temp: Whether to fix the temperature
        """
        super().__init__()
        
        self.size = size 
        self.log_alpha = nn.Parameter(torch.ones(size) * init_value)

        
        # Distribution parameters
        self.beta = beta  # Temperature
        self.gamma = gamma  # Stretch left
        self.zeta = zeta  # Stretch right
    
    
    def sample_gate(self, reuse_u=None):
        """
        Sample from the Hard-Concrete distribution
        
        Args:
            reuse_u: Reuse previously generated uniform samples (for consistency in forward/backward)
            
        Returns:
            gate: Sampled gate values
            u: Uniform samples used (to enable reuse)
        """
        # Calculation based on Hard-Concrete/L0 regularization paper
        if self.training:
            # Generate noise from uniform distribution
            if reuse_u is None:
                u = torch.rand(self.size, device=self.log_alpha.device)
            else:
                u = reuse_u
            
            # Transform u with sigmoid and log-alpha
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha) / self.beta)
            
            # Apply stretching to get values potentially outside [0, 1]
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            
            # Hard-sigmoid to get binary-like outputs
            z = torch.clamp(s_stretched, 0, 1)
            
            return z, u
        else:
            # During evaluation, use expected value (deterministic)
            s = torch.sigmoid(self.log_alpha / self.beta)
            s_stretched = s * (self.zeta - self.gamma) + self.gamma
            z = torch.clamp(s_stretched, 0, 1)
            return z, None
    
    def forward(self, reuse_u=None):
        gate, u = self.sample_gate(reuse_u)
        return gate, u
    
    def get_expected_L0_norm(self):
        """
        Calculate the expected L0 norm of the gate (number of non-zero elements)
        Used for regularization and monitoring sparsity
        """
        s = torch.sigmoid(self.log_alpha / self.beta)
        
        # Expected L0 norm (probability of a gate being active)
        expected_l0 = torch.sum(s * (self.zeta - self.gamma) + self.gamma > 0)
        
        return expected_l0
    
    def get_expected_gates(self):
        """
        Calculate the expected value of each gate
        Used for monitoring and deterministic evaluation
        """
        s = torch.sigmoid(self.log_alpha / self.beta)
        s_stretched = s * (self.zeta - self.gamma) + self.gamma
        expected_gates = torch.clamp(s_stretched, 0, 1)
        
        return expected_gates
    
    def count_active_gates(self, threshold=0.5):
        """
        Count gates that are active (above threshold)
        Used for monitoring sparsity
        """
        expected_gates = self.get_expected_gates()
        return torch.sum(expected_gates > threshold).item()
    
class LinearWithSALoRA(LinearWithLoRA):
    def __init__(self, lora_config: LoRAConfig, init_r, target_r):
        super().__init__(lora_config)
        self.lora_scaler = lora_config.lora_scaler
        self.lora_rank = init_r
        self.target_rank = target_r
        
    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        weight_a = self.weight_a.to(
            self._get_lora_dtype()
        )
        weight_b = self.weight_b.to(
            self._get_lora_dtype()
        )

        gate, u = self.hc_gate(self.last_u)
        self.last_u = u
        self.last_gate = gate

        ranknum = self.ranknum + 1

        lora_result = F.linear(
            F.linear(self.lora_dropout(x), weight_a * gate),
            weight_b,
            ).to(result.dtype)

        return result + lora_result * self.lora_scaler / ranknum
    
    def _compute_lora(self):
        if self.has_lora_weights:
            weight_a = self.weight_a.to(
                self._get_lora_dtype()
            )
            weight_b = self.weight_b.to(
                self._get_lora_dtype()
            )
            
            gate = self.hc_gate.get_expected_gates()
            
            gated_weight_a = weight_a * gate
            
        ranknum = self.lora_rank + 1
        lora_result = F.linear(gated_weight_a, weight_b)
        lora_weight = lora_result * self.lora_scaler / ranknum
        return lora_weight
    
    def init_lora_weights(self):
        # called by __init__ in LinearWithLoRA
        dtype = self._get_lora_dtype()
        
        # Initialize A and B matrices with structural awareness
        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
        self.hc_gate = HardConcreteGate(size=(self.lora_rank, 1))
        self.last_u = None
        self.last_gate = None
        
        nn.init.orthogonal_(self.weight_a)
        nn.init.orthogonal_(self.weight_b)
        
        requires_grad = not self.quant
        
        # All ranks are activated at the very beginging of training.
        self.lamda = nn.Parameter(torch.randn(1), requires_grad=requires_grad)
        
        nn.init.normal_(self.lamda, mean=1.0, std=0.1)
        
        self.ranknum = nn.Parameter(torch.randn(1), requires_grad=False)
        self.ranknum.data.fill_(float(self.lora_rank))
        self.ranknum.requires_grad = False
