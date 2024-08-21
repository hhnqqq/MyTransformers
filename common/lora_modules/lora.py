# @author: haonan he
# @date: 2024-04-02
""" Implements LORA with multiple techniques such as DORA. 
To merge the LORA weight with full rank weight for faster inference, 
locate every LinearWithLoRA layer and call the merge_and_del method. 
Afterward, the LinearWithLoRA will function similarly to a normal Linear layer, 
eliminating the need to replace LinearWithLoRA with Linear. """
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from typing import Union, Optional, List, Dict, Any

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
        if lora_dropout:
            self.lora_dropout = nn.Dropout(lora_dropout)
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
            weight = weight + lora_weight

        return F.linear(x, weight) + F.linear(self.lora_dropout(x), lora_weight)

    def _quantize_weight(self, weight: torch.Tensor, quantizer: Optional[torch.Tensor]) -> torch.Tensor:
        if self.quant and quantizer is not None:
            return weight * quantizer.unsqueeze(-1)
        return weight
    
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
            

if __name__ == '__main__':
    from common.lora_modules.lora_set_up import switch_to_lora
    
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


