# @author: haonan he
# @date: 2024-08-21
""" Implements MELORA"""

import re
from common.lora_modules.lora import *

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
