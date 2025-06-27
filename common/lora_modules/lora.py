# @author: haonan he
# @date: 2024-04-02
""" Implements LORA with powerful methods like merge_and_reset. 
To merge the LORA weight with full rank weight for faster inference, 
locate every LinearWithLoRA layer and call the merge_and_del method. 
Afterward, the LinearWithLoRA will function similarly to a normal Linear layer, 
eliminating the need to replace LinearWithLoRA with Linear. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class LoRAConfig:
    in_features: int
    out_features: int
    bias: bool = False
    lora_rank: int = 4
    lora_scaler: float = 32.0
    lora_dropout: Optional[float] = None
    quant: bool = False
    weight_a_init_method: Optional[str] = None
    weight_b_init_method: Optional[str] = None
    run_lora_in_fp32: bool = False

class LinearWithLoRA(nn.Linear):
    def __init__(
        self,
        lora_config: LoRAConfig
    ):
        """
        Initialize the LinearWithLoRA layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            lora_rank (int, optional): Rank of LoRA decomposition. Default is 4.
            lora_scaler (float, optional): Scaler for LoRA weights. Default is 32.0.
            quant (bool, optional): Whether to apply weight quantization. Default is False.
            weight_a_init_method (str, optional): The init method for weight_a.
            weight_b_init_method (str, optional): The init method for weight_b.
            run_lora_in_fp32 (bool): Whether to keep lora weight in fp32 regardless of dtype of forzen weight. (Defualt setting in peft's lora implementation.)
        """
        super().__init__(lora_config.in_features, lora_config.out_features, bias=lora_config.bias)
        self.lora_rank = lora_config.lora_rank
        self.lora_scaler = lora_config.lora_scaler / lora_config.lora_rank
        self.quant = lora_config.quant
        self.weight_a_init_method = lora_config.weight_a_init_method
        self.weight_b_init_method = lora_config.weight_b_init_method
        self.run_lora_in_fp32 = lora_config.run_lora_in_fp32
        self.disable_lora = False
            
        if lora_config.lora_dropout:
            self.lora_dropout = nn.Dropout(lora_config.lora_dropout)
        else:
            self.lora_dropout = nn.Identity()
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The origin weight of Linear layer.
        weight = self._quantize_weight(self.weight, self.weight_quantizer)
        result = F.linear(x, weight, self.bias)
        if self.disable_lora or not self.has_lora_weights:
            return result
        else:
            # Make sure the input of lora have same dtype with lora weights.
            x = x.to(self._get_lora_dtype())
            return self._lora_forward(x, result)

    def _lora_forward(self, x: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        # If self.run_lora_in_fp32, then the dtype of lora_result will be fp32.
        weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(self._get_lora_dtype())
        weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(self._get_lora_dtype())
        
        lora_result = F.linear(F.linear(self.lora_dropout(x), weight_a), weight_b).to(result.dtype)
        return result + self.lora_scaler * lora_result
    
    def _quantize_weight(self, weight: torch.Tensor, quantizer: Optional[torch.Tensor]) -> torch.Tensor:
        if self.quant and quantizer is not None:
            return weight * quantizer.unsqueeze(-1)
        return weight
    
    def _get_lora_dtype(self):
        dtype = torch.int8 if self.quant else self.weight.dtype
        if self.run_lora_in_fp32:
            dtype = torch.float32
        return dtype
    
    def init_lora_weights(self):
        # Defualt is fp32 when LinearWithLora init.
        dtype = self._get_lora_dtype()
        requires_grad = not self.quant

        self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=dtype), requires_grad=requires_grad)
        self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=dtype), requires_grad=requires_grad)

        if self.quant:
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

        self._init_weight('weight_a')
        self._init_weight('weight_b')

    def std_normalization(self):
        if not torch.all(torch.eq(self.weight_a, 0)):
            self.weight_a.data = self.weight_a.data / self.weight_a.data.std()
        if not torch.all(torch.eq(self.weight_b, 0)):
            self.weight_b.data = self.weight_b.data / self.weight_b.data.std()
            
    def _compute_lora_weight(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer).to(self._get_lora_dtype())
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer).to(self._get_lora_dtype())
            lora_weight = self.lora_scaler * torch.matmul(weight_b, weight_a)
            return lora_weight.to(self.weight.dtype)
        
    def _merge_lora(self) -> bool:
        # Merge the lora weight into full rank weight if possible.
        if self.has_lora_weights:
            # Compute lora weight.
            lora_weight = self._compute_lora_weight()
            self.weight.data += lora_weight
            return True
        return False
    
    def _unmerge_lora(self) -> bool:
        if self.has_lora_weights:
            lora_weight = self._compute_lora_weight()
            self.weight.data -= lora_weight
            return True
        return False

    def merge_and_reset(self, new_rank: Optional[int] = None):
        # If there is lora weight and it has been successfully merged, reinitialize the lora weight:
        if new_rank is not None:
            self.merge_and_del()
            self.lora_rank = new_rank
            self.init_lora_weights()
        else:
            if self._merge_lora():
                self._init_weight('weight_a')
                self._init_weight('weight_b')
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
            self.init_lora_weights()

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
    def has_lora_weights(self):
        has_attr = hasattr(self, 'weight_a') and hasattr(self, 'weight_b')
        if has_attr:
            is_not_None = self.weight_a is not None and self.weight_b is not None
        return has_attr and is_not_None

    def _init_weight(self, weight_name: str):
        weight = getattr(self, weight_name)
        init_method = getattr(self, f"{weight_name}_init_method")
        init_kwargs = self.get_weight_init_kwargs(weight_name, init_method)
        if init_method and 'method' not in init_kwargs.keys():
            init_kwargs['method'] = init_method
        self.get_weight_init_method(**init_kwargs)(weight)

    def get_weight_init_kwargs(self, weight_name: str, method: Optional[str] = None) -> Dict[str, Any]:
        init_configs = {
            'weight_a': {None:{'std': 1 / (self.in_features ** 0.5), 'mean': 0.0},
                         'normal':{'std': 0.02, 'mean': 0.0},
                         'kaiming':{'a':5**0.5, 'mode':'fan_in'}},
            'weight_b': {None:{'method':'zeros'},
                         'guassian':{'std': 1 / (self.lora_rank ** 0.5), 'mean': 0.0},
                         'unit':{'std': 1 / (self.lora_rank ** 0.5), 'mean': 0.0},
                         'normal':{'std': 0.02, 'mean': 0.0},
                         'kaiming':{'a':5**0.5, 'mode':'fan_in'}}
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
            'normal': partial(nn.init.normal_, std=init_kwargs.get('std', 1), 
                            mean=init_kwargs.get('mean', 0)),
            'unit': partial(nn.init.normal_, std=init_kwargs.get('std', 1), 
                            mean=init_kwargs.get('mean', 0)),
            'orthogonal': nn.init.orthogonal_,
            "uniform": partial(nn.init.uniform_, a=init_kwargs.get('a', 0), b=init_kwargs.get('b', 1))
        }

        if method in init_methods:
            return init_methods[method]
        
        raise ValueError(f"Unknown initialization method: {method}")
    
    def print_details(self) -> None:
        print(f"{self.__class__.__name__} Layer: in_features={self.in_features}, out_features={self.out_features}")
        print(f"Lora Enabled: {self.has_lora_weights}, LoRA Rank: {self.lora_rank}, Quantized: {self.quant}")
            

def find_lora_names(n):
    for substring in ['weight_a', 'weight_b']:
        if substring in n:
            return substring
    return ""