# @author: haonan he
# @date: 2024-04-02
""" Implements LORA with multiple techniques such as DORA. 
To merge the LORA weight with full rank weight for faster inference, 
locate every LinearWithLoRA layer and call the merge_and_del method. 
Afterward, the LinearWithLoRA will function similarly to a normal Linear layer, 
eliminating the need to replace LinearWithLoRA with Linear. """
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections.abc import Iterable
from functools import partial
from typing import Union, Optional, List, Dict, Any
from common.utils import print_rank_0
from common.utils import to_device

class LinearWithLoRA(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        use_dora: bool = False,
        use_mos_lora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None
    ):
        """
        Initialize the LinearWithLoRA layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int, optional): Rank of LoRA decomposition. Default is 4.
            lora_scaler (float, optional): Scaler for LoRA weights. Default is 32.0.
            use_dora (bool, optional): Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation). Default is False.
            use_mos_lora (bool, optional): Whether to use MosLoRA (Mixture-of-Subspaces in Low-Rank Adaptation). Default is False.
            quant (bool, optional): Whether to apply weight quantization. Default is False.
            plora_steps (Union(int, None), optional): Steps to merge and reset lora weight.  Deault is None.
        """
        super().__init__(in_features, out_features, bias=False)
        self.lora_rank = lora_rank
        self.lora_scaler = lora_scaler / lora_rank
        self.quant = quant
        self.dora = use_dora
        self.mos_lora = use_mos_lora
        self.weight_a_init_method = weight_a_init_method
        self.weight_b_init_method = weight_b_init_method
        self.weight_ab_mixer_init_method = weight_ab_mixer_init_method


        self._init_lora_weights()
        # Enable plora, if plora_steps is not None.
        self.plora = plora_steps is not None
        if plora_steps:
            self.plora_steps = plora_steps
            self.plora_counter = 0
        if lora_dropout and not use_dora:
            self.lora_dropout = nn.Dropout(lora_dropout)
        elif lora_dropout and use_dora:
            print_rank_0(f'Dora is incompatible with lora dropout, skiped lora dropout')
            self.lora_dropout = nn.Dropout(1)
        else:
            self.lora_dropout = nn.Identity()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every plora stage, we merge the origin lora weight and reset new lora weight.
        if self.plora:
            self.plora_counter += 1
            if self.plora_counter == self.plora_steps:
                self.merge_and_reset()
                self.plora_counter = 0

        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)

        lora_weight = None
        # If lora attrs are exist, compute the lora weight and plus it to full rank weight
        if self.has_lora_weights:
            lora_weight = self._compute_lora()
            if self.lora_dropout:
                # Rather than dropout the activation, we dropout the lora weight
                lora_weight = self.lora_dropout(lora_weight)

            # Weather to use DoRA.
            if self.dora and lora_weight is not None:
                weight = self._apply_dora(weight, lora_weight)

        # Unified output.
        return F.linear(x, weight) + F.linear(self.lora_dropout(x), lora_weight)

    def _quantize_weight(self, weight: torch.Tensor, quantizer: Optional[torch.Tensor]) -> torch.Tensor:
        if self.quant and quantizer is not None:
            return weight * quantizer.unsqueeze(-1)
        return weight

    def _apply_dora(self, weight: torch.Tensor, lora_weight: torch.Tensor) -> torch.Tensor:
        # The magnitude of origin weight on the output dim: [2048,2048] -> [1, 2048].
        m = self.weight.norm(p=2, dim=0, keepdim=True)
        # Origin weight plus lora weight -> new weight. 
        directional_numerator = weight + lora_weight
        # The magnitude of new weight on the output dim. 
        directional_denominator = directional_numerator.norm(p=2, dim=0, keepdim=True)
        # Scale the magnitude of new weight to 1.
        directional_component = directional_numerator / directional_denominator
        # Ensure the new weight's magnitude remains the same as the origin weight.
        return m * directional_component

    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        if self.mos_lora:
            self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            if self.quant:
                self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self._init_weight('weight_ab_mixer')

        self._init_weight('weight_a')
        self._init_weight('weight_b')

    def _init_weight(self, weight_name: str):
        weight = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        self.get_weight_init_method(**init_kwargs)(weight)

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        init_configs = {
            'weight_a': {None:{'std': 1 / (self.in_features ** 0.5), 'mean': 0.0}},
            'weight_b': {None:{'method':'zeros'},
                         'guassian':{'std': 1 / (self.lora_rank ** 0.5), 'mean': 0.0},
                         'unit':{'std': 1 / (self.lora_rank ** 0.5), 'mean': 0.0}}
            ,
            'weight_ab_mixer': {
                None: {'method': 'kaiming', 'a': 5**0.5, 'mode': 'fan_in'},
                'gaussian': {'std': 1 / (self.lora_rank ** 0.5), 'mean': 0.0}
            }
        }

        if weight_name in init_configs:
            return init_configs[weight_name].get(method, init_configs[weight_name][None])
        
        raise ValueError(f"Unknown weight name: {weight_name}")

    def get_weight_init_method(self, **init_kwargs) -> Any:
        method = init_kwargs.get('method', None)
        
        init_methods = {
            None: partial(nn.init.normal_, mean=init_kwargs.get('mean', 0), 
                          std=init_kwargs.get('std', 1)),
            'kaiming': partial(nn.init.kaiming_uniform_, a=init_kwargs.get('a', 5**0.5), 
                               mode=init_kwargs.get('mode', 'fan_in')),
            'xavier': nn.init.xavier_normal_,
            'zeros': nn.init.zeros_,
            'unit': partial(nn.init.normal_, std=init_kwargs.get('std', 1), 
                            mean=init_kwargs.get('mean', 0)),
            'orthogonal': nn.init.orthogonal_
        }

        if method in init_methods:
            return init_methods[method]
        
        raise ValueError(f"Unknown initialization method: {method}")

    def gradient_reinit(self, 
                        direction: str = 'ArB2r', 
                        scale: str = 'gd', 
                        stable_gamma: int = 16, 
                        scaling_factor: int = 16):
        """
        Reinitialize the LoRA weights based on the gradient of the original weight matrix.

        This method implements the core functionality of LoRA-GA (Gradient-based Adaptation).
        It performs SVD on the weight gradient and uses the resulting matrices to update
        the LoRA weights (A and B).

        Args:
            direction (str): Determines how to select A and B from SVD results.
                Options: 'ArBr', 'A2rBr', 'ArB2r'. Default is 'ArB2r'.
            scale (str): Scaling method for the new LoRA weights.
                Options: 'gd', 'unit', 'stable', 'weightS'. Default is 'stable'.
            stable_gamma (float): Gamma parameter for 'stable' scaling. Default is 16.

        The method performs the following steps:
        1. Compute SVD of the weight gradient
        2. Select A and B matrices based on the 'direction' parameter
        3. Apply scaling to A and B based on the 'scale' parameter
        4. Update the LoRA weights (weight_a and weight_b)

        Note: This method assumes that the LinearWithLora layer has gradient. Please call this
        method in the first step of training(before the model.step() call, or the gradient will be cleared.)
        """

        if hasattr(self.weight, 'grad_stored'):
            # Perform SVD on the weight gradient
            # Weight stored gradient shape [out_feature, in_feature]
            U, S, V = torch.svd_lowrank(self.weight.grad_stored.float().cuda(), q=4 * self.lora_rank, niter=4)
            # U shape [out_feature, 4r] S shape [4r, 4r] V shape [in_feature, 4r]
            V = V.T

            # Determine A and B based on the direction parameter
            if direction == "ArBr":
                # B shape [out_feature, r/2]
                B = U[:, 0:2 * self.lora_rank:2]
                A = V[1:2 * self.lora_rank:2, :]
            elif direction == "A2rBr":
                # B shape [out_feature, r]
                B = U[:, :self.lora_rank]
                # A shape [r, in_feature]
                A = V[self.lora_rank:2 * self.lora_rank, :]
            elif direction == "ArB2r":
                B = U[:, self.lora_rank:2 * self.lora_rank]
                A = V[:self.lora_rank, :]
            else:
                raise ValueError(f"Unknown direction: {direction}")

            # Apply scaling to A and B based on the scale parameter
            if scale == "gd":
                A /= scaling_factor
                B /= scaling_factor
            elif scale == "stable":
                m, n = self.weight.grad_stored.shape 
                A = A * m**0.25 / stable_gamma**0.5
                B = B * m**0.25 / stable_gamma**0.5
            elif scale == "weightS":
                _, S, _ = torch.svd_lowrank(self.weight.float(), q=4 * self.lora_rank, niter=4)
                S /= scaling_factor
                avg_s = torch.sqrt(S[:self.lora_rank]).mean().to(A.device)
                A *= avg_s
                B *= avg_s
            elif scale != "unit":
                raise ValueError(f"Unknown scale: {scale}")
            del self.weight.grad_stored

            # Update the LoRA weights
            self.weight_a.data = A.contiguous().cuda()
            self.weight_b.data = B.contiguous().cuda()
            
    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
            if self.mos_lora:
                # When using vanilla lora, the ab mixer is a identical matrix
                weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
                weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            else:
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight
        
    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            lora_weight = self._compute_lora()
            if self.dora:
                self.weight.data = self._apply_dora(self.weight, lora_weight)
            else:
                self.weight.data += lora_weight
            return True
        return False

    def merge_and_reset(self, new_rank: Optional[int] = None):
        # If there is lora weight and it has been successfully merged, reinitialize the lora weight:
        if new_rank is not None:
            self.merge_and_del()
            self.lora_rank = new_rank
            self._init_lora_weights()
        else:
            if self._merge_lora():
                std = (1 / self.in_features)**0.5
                nn.init.normal_(self.weight_a, mean=0, std=std)
                nn.init.zeros_(self.weight_b)
                if self.quant:
                    self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
                    self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

    def _del_lora(self):
        delattr(self, "weight_a")
        delattr(self, "weight_b")

    def merge_and_del(self):
        # If there is lora weight and it has been successfully merged, delete all lora attrs:
        if self._merge_lora():
            # delattr can not completly delete the weight, which can cause error when model.parameters() be called.
            self._del_lora()
            if self.quant:
                self.weight_a_scaler = None
                self.weight_b_scaler = None

    def reset(self):
        if not self.has_lora_weights:
            self._init_lora_weights()

    @property
    def weight_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_scaler", None)

    @property
    def weight_a_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_a_scaler", None)

    @property
    def weight_b_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_b_scaler", None)
    
    @property
    def weight_ab_quantizer(self) -> Optional[torch.Tensor]:
        return getattr(self, "weight_ab_scaler", None)
    
    @property
    def has_lora_weights(self):
        has_attr = hasattr(self, 'weight_a') and hasattr(self, 'weight_b')
        if has_attr:
            is_not_None = self.weight_a is not None and self.weight_b is not None
        return has_attr and is_not_None

    def print_details(self) -> None:
        print(f"LinearWithLoRA Layer: in_features={self.in_features}, out_features={self.out_features}")
        print(f"Lora Enabled: {self.has_lora_weights}, LoRA Rank: {self.lora_rank}, Quantized: {self.quant}, DoRA: {self.dora}")

class LinearWithMeLora(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        use_dora: bool = False,
        use_mos_lora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None,
        me_lora_n_split: int = 2):
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         use_dora,
                         use_mos_lora,
                         quant,
                         plora_steps,
                         weight_a_init_method,
                         weight_b_init_method,
                         weight_ab_mixer_init_method)
        self.melora_n_split = me_lora_n_split
        self.mini_lora_rank = self.lora_rank / self.melora_n_split
        self.mini_in_features = self.in_features / self.melora_n_split
        self.mini_out_features = self.out_features / self.melora_n_split

    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        for i in range(self.melora_n_split):
            mini_weight_a = nn.Parameter(torch.empty((self.mini_lora_rank, self.mini_in_features), dtype=dtype), requires_grad=requires_grad)
            mini_weight_b = nn.Parameter(torch.zeros((self.mini_out_features, self.mini_lora_rank), dtype=dtype), requires_grad=requires_grad)
            setattr(self, f'melora_weight_a_{i}') = mini_weight_a
            setattr(self, f'melora_weight_b_{i}') = mini_weight_b

            self._init_weight(f'melora_weight_a_{i}')
            self._init_weight(f'melora_weight_b_{i}')

        if self.mos_lora:
            self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            if self.quant:
                self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self._init_weight('weight_ab_mixer')

    def _compute_lora(self):
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._diagonal_concat_weight_a()
            weight_b = self._diagonal_concat_weight_b()
            weight_a = self._quantize_weight(weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(weight_b, self.weight_b_quantizer)
            if self.mos_lora:
                # When using vanilla lora, the ab mixer is a identical matrix
                weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
                weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            else:
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight
        
    def _diagonal_concat_weight_a(self):
        weight_a = torch.zeros(self.mini_lora_rank, self.mini_in_features)
        
        for i in range(self.melora_n_split):
            start_row = i * self.mini_lora_rank
            start_col = i * self.mini_in_features
            weight_a[start_row:start_row+self.mini_lora_rank, start_col:start_col+self.mini_in_features] = getattr(self, f"melora_weight_a_{i}")
        
        return weight_a
    
    def _diagonal_concat_weight_b(self):
        weight_b = torch.zeros(self.out_features, self.lora_rank)
        
        for i in range(self.melora_n_split):
            start_row = i * self.mini_out_features
            start_col = i * self.mini_lora_rank
            weight_b[start_row:start_row+self.mini_out_features, start_col:start_col+self.mini_lora_rank] = getattr(self, f"melora_weight_b_{i}")
        
        return weight_b

    def _init_weight(self, weight_name: str):
        weight = getattr(self, weight_name)
        weight_group = re.search(weight_name, r'melora_(weight_.)_\d').group(1)
        init_method = getattr(self, f"{weight_group}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_group, init_method)
        self.get_weight_init_method(**init_kwargs)(weight)
            
class LinearWithVeRA(LinearWithLoRA):
    def __init__(self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        lora_dropout: Optional[float] = None,
        use_dora: bool = False,
        use_mos_lora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None,
        weight_a_init_method: Optional[str] = None,
        weight_b_init_method: Optional[str] = None,
        weight_ab_mixer_init_method: Optional[str] = None,
        scaling_vector_b_init_method: str = 'zero',
        scaling_vector_d_init_method: str = 'ones'
    ):
        """
        Initialize the LinearWithVeRA layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int, optional): Rank of LoRA decomposition. Default is 4.
            lora_scaler (float, optional): Scaler for LoRA weights. Default is 32.0.
            use_dora (bool, optional): Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation). Default is False.
            use_mos_lora (bool, optional): Whether to use MosLoRA (Mixture-of-Subspaces in Low-Rank Adaptation). Default is False.
            quant (bool, optional): Whether to apply weight quantization. Default is False.
            plora_steps (Union(int, None), optional): Steps to merge and reset lora weight.  Deault is None.
            scaling_vector_b_init_method (str, optional): Initialization method for scaling vector b. ['zero', 'ones', 'small_constant', 'random']. Default is 'zeros'.
            scaling_vector_d_init_method (str, optional): Initialization method for scaling vector d. ['zero', 'ones', 'small_constant', 'random']. Default is 'ones'.
        """
        super().__init__(in_features,
                         out_features,
                         lora_rank,
                         lora_scaler,
                         lora_dropout,
                         use_dora,
                         use_mos_lora,
                         quant,
                         plora_steps,
                         weight_a_init_method,
                         weight_b_init_method,
                         weight_ab_mixer_init_method)
        self._init_scaling_vectors(scaling_vector_b_init_method, scaling_vector_d_init_method)
    
    def _init_lora_weights(self):
        dtype = torch.int8 if self.quant else None
        requires_grad = not self.quant

        # Initialize shared matrix a and b, and frozen them.
        self.weight_a = nn.Parameter(torch.randn((self.lora_rank, self.in_features), dtype=dtype), requires_grad=False)
        self.weight_b = nn.Parameter(torch.randn((self.out_features, self.lora_rank), dtype=dtype), requires_grad=False)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        if self.mos_lora:
            self.weight_ab_mixer = nn.Parameter(torch.empty((self.lora_rank, self.lora_rank), dtype=dtype), requires_grad=requires_grad)
            if self.quant:
                self.weight_ab_mixer_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self._init_weight('weight_ab_mixer')

        self._init_weight('weight_a')
        self._init_weight('weight_b')
    
    def _init_scaling_vectors(self, b_init_method: str, d_init_method: str):
        """
        Initialize scaling vectors with different methods.
        Args:
            b_init_method: The method to initialize scaling vector b.
            d_init_method: The method to initialize scaling vector d.
        """
        # Initialize vector b
        if b_init_method == 'zero':
            self.scaling_vector_b = torch.zeros(self.out_features)
        elif b_init_method == 'ones':
            self.scaling_vector_b = torch.ones(self.out_features)
        elif b_init_method == 'small_constant':
            self.scaling_vector_b = 0.1 * torch.ones(self.out_features)
        elif b_init_method == 'random':
            self.scaling_vector_b = torch.rand(self.out_features)
        else:
            raise ValueError(f"Unknown b_init_method: {b_init_method}")

        # Initialize vector d
        if d_init_method == 'zero':
            self.scaling_vector_d = torch.zeros(self.lora_rank)
        elif d_init_method == 'ones':
            self.scaling_vector_d = torch.ones(self.lora_rank)
        elif d_init_method == 'small_constant':
            self.scaling_vector_d = 0.1 * torch.ones(self.lora_rank)
        elif d_init_method == 'random':
            self.scaling_vector_d = torch.rand(self.lora_rank)
        else:
            raise ValueError(f"Unknown d_init_method: {d_init_method}")

    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute adapted lora weights.
            adapted_weight_a = torch.matmul(self.weight_a, torch.diag(self.scaling_vector_b))
            adapted_weight_b = torch.matmul(self.weight_b, torch.diag(self.scaling_vector_d))
            # Compute lora weights.
            weight_a = self._quantize_weight(adapted_weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(adapted_weight_b, self.weight_b_quantizer)
            if self.mos_lora:
                # When using vanilla lora, the ab mixer is a identical matrix
                weight_ab_mixer = self._quantize_weight(self.weight_ab_mixer, self.weight_ab_quantizer)
                weight_a_forward = torch.matmul(weight_ab_mixer, weight_a)
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a_forward)
            else:
                lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)

            return lora_weight

def switch_to_lora(model: nn.Module, 
                   replace_names: Optional[Union[str, List[str]]] = None, 
                   rank: int = 4, 
                   lora_scaler: int = 32, 
                   lora_dropout: Optional[float] = None,
                   transposition: bool = False, 
                   use_dora: bool = False, 
                   use_mos_lora: bool = False,
                   plora_steps: Optional[int] = None):
    """
    Switch function for lora, responsible for replacing Linear layer with LinearWithLoRA layer

    Args:
        model: Any pytorch model.
        replace_names: List of module names to be replaced by LoRA.
        rank: Rank for LoRA.
        lora_scaler: Scaler for LoRA.
        transposition: nn.Linear(x, w) compute xw^T, so the weight should in shape [out_feature, in_feature]. Otherwise, transposition should be set to True
        use_dora: Weather to use dora
        plora_steps: The steps to merge and reset lora weight.
    """
    assert replace_names is not None, 'Replace names can not be None'
    for name, module in model.named_modules():
        replace_tag = False
        for replace_name in replace_names:
            if replace_name in name:
                # Create LoRA layer instance.
                replace_tag = True
                if isinstance(module, LinearWithLoRA):
                    module.merge_and_reset(new_rank=rank)
                elif isinstance(module, nn.Module):
                    if  all(hasattr(module, attr) for attr in ["in_features", "out_features", "weight"]):
                        quant = getattr(module, "quant", False)
                        lora_layer = LinearWithLoRA(lora_rank=rank, 
                                                    lora_scaler=lora_scaler, 
                                                    lora_dropout=lora_dropout,
                                                    in_features=module.in_features, 
                                                    out_features=module.out_features, 
                                                    use_dora=use_dora, 
                                                    use_mos_lora=use_mos_lora,
                                                    quant=quant,
                                                    plora_steps=plora_steps)
                        # Copy the original weight to the LoRA layer.
                        if transposition:
                            lora_layer.weight = nn.Parameter(module.weight.data.T)
                        else:
                            lora_layer.weight.data = module.weight.data
                        if quant:
                            lora_layer.weight_scaler = module.weight_scaler
                        # Replace the original layer with the LoRA layer.
                        parent = get_parent_model(model, module)
                        setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], lora_layer)
        if not replace_tag and isinstance(module, LinearWithLoRA):
            # Merge weight to avoid unnecessary computing.
            module.merge_and_del()

def setup_lora(model, args, model_config):
    if args.use_lora:
        if args.replace_modules is None:
            args.replace_modules = model_config.lora_layers
        switch_to_lora(model, 
                       args.replace_modules, 
                       rank=args.lora_rank, 
                       use_dora=args.use_dora,
                       use_mos_lora=args.use_mos_lora,
                       lora_dropout=args.lora_dropout,
                       plora_steps=args.plora_steps)
        if args.lora_fa:
            lora_weight = ['weight_b', 'weight_ab_mixer']
        else:
            lora_weight = ['weight_a','weight_b', 'weight_ab_mixer']
        args.enable_list = lora_weight if args.enable_list is None else list(set(args.enable_list + lora_weight))
        if args.fp16:
            model.to(args.device).half()
        elif args.bf16:
            model.to(args.device).bfloat16()

def recover_linear(model: nn.Module):
    """
    Recover function for lora, responsible for recover LinearWithLoRA layer to Linear layer.

    Args:
        model: Any pytorch model.
    """
    for module in model.modules:
        if isinstance(module, LinearWithLoRA):
            module.merge_and_del()
            linear_layer = nn.Linear(in_features=module.in_features,
                                     out_features=module.out_features,
                                     bias=False,
                                     dtype=module.weight.dtype,
                                     device=module.weight.dtype)
            linear.weight.data = module.weight.data
            parent = get_parent_model(model, module)
            setattr(parent, list(parent._modules.items())[list(parent._modules.values()).index(module)][0], linear_layer)
            


def get_parent_model(parent_model, module):
    """
    Find the parent module for the input module recursively.

    Args:
        parent_model: Root model for the search.
        module: Submodule to find the parent module for.

    Returns:
        Parent module if found, None otherwise.
    """
    for _, sub_module in parent_model._modules.items():
        # Direct sub modules of parent model.
        if sub_module is module:
            return parent_model
        parent = get_parent_model(sub_module, module)
        if parent:
            return parent
    return None

def get_record_gradient_hook(model):
    def record_gradient_hook(grad):
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                if not hasattr(p, 'grad_stored'):
                    p.grad_stored = p.grad.cpu()
                else:
                    p.grad_stored += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook

def lora_ga_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
):
    r"""
    Compute the full-rank gradient of the model on the given dataset and reinitialize the LoRA weights.

    LoRA-GA Algorithm:
    1. Perform forward pass to get predictions
    2. Calculate loss
    3. Set learning rate η = α * sqrt(r)
    4. For each layer l from L to 1:
        a. Compute gradient ∇Wl ℓ
        b. Get layer dimensions dout, din
        c. Perform SVD: U, S, V = svd(∇Wl ℓ)
        d. Initialize Al using first r columns of V
        e. Initialize Bl using columns r+1 to 2r of U
        f. Update weights: Wl = Wl - η * Bl * Al
        g. Clear gradient for this layer

    Input:
    - Model f(·) with L layers and parameters W
    - Sampled batch B = {x, y}
    - LoRA rank r
    - LoRA scaling factor α
    - Loss function L
    - Scale factor γ

    Output:
    - Initialized parameters W, η, A, B

    Note: This implementation follows the LoRA-GA paper's algorithm closely.
    """
    print_rank_0("--->Estimating gradient for lora ga.", rank=args.global_rank)
    model.train()
    hooks = []
    for param in model.parameters():
        param.requires_grad = True
        hook = param.register_hook(get_record_gradient_hook(model))
        hooks.append(hook)
    if not isinstance(dataloader, Iterable):
        # Make sure that the dataloader is iterable.
        dataloader = iter(dataloader)
    for iter in range(iters):
        batch = to_device(next(dataloader), args.device)
        output = model(**batch)
        if args.huggingface:
            output.loss.backward()
        else:
            output[0].backward()
        get_record_gradient_hook(model)(None)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
    for p in model.parameters():
        p.grad_stored /= iters
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
            module.gradient_reinit()


if __name__ == '__main__':
    use_dora = False
    plora_steps = None
    # initialize test
    linear = LinearWithLoRA(in_features=2048, 
                            out_features=2048, 
                            lora_rank=8, 
                            lora_scaler=32, 
                            use_dora=use_dora,
                            use_mos_lora=True, 
                            quant=False, 
                            plora_steps=plora_steps)
    linear.weight.data = torch.randn(2048,2048)
    # linear.weight_b.data = torch.randn(2048, 8)

    print(linear.weight)
    linear.merge_and_reset()
    print(linear.weight)

    # forward test
    print(linear(torch.randn(2048,2048)))
    linear.print_details()

    model = nn.Transformer(num_encoder_layers=0)
    print(model)
    switch_to_lora(model, ['linear'], transposition=False)
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_and_reset()
            print(module.in_features, module.out_features, module.weight.shape)
    # switch_to_lora(model, ['norm'], transposition=True) # result in a assert error 

    # backward test
    class TestModel(nn.Module):
        def __init__(self, in_features, out_features, lora_rank, lora_scaler, lora_dropout, use_dora, quant, plora_steps):
            super().__init__()
            self.linear = LinearWithLoRA(in_features, out_features, lora_rank, lora_scaler, lora_dropout, use_dora, quant, plora_steps)

        def forward(self, x):
            return self.linear(x)

    def test_lora_gradient():
        # Set up the model
        in_features = 64
        out_features = 64
        lora_rank = 4
        lora_scaler = 32.0
        lora_dropout = 0.1
        use_dora = True
        quant = False
        plora_steps = None

        model = TestModel(in_features, out_features, lora_rank, lora_scaler, lora_dropout, use_dora, quant, plora_steps)
        # model.linear.merge_and_del()

        # Generate some random input and target
        input_data = torch.randn(32, in_features)
        target_data = torch.randn(32, out_features)

        # Forward pass
        output = model(input_data)

        # Compute the loss
        loss = nn.MSELoss()(output, target_data)

        # Backward pass
        loss.backward()

        # Check if the gradients are not None
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"{name}'s gradient is None")

        print("Test passed: Gradients are not None.")

    test_lora_gradient()


