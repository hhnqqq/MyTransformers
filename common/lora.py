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

from typing import Union, Optional


class LinearWithLoRA(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_rank: int = 4,
        lora_scaler: float = 32.0,
        use_dora: bool = False,
        quant: bool = False,
        plora_steps: Union[int, None] = None
    ):
        """
        Initialize the LinearWithLoRA layer.
        param in_features (int): Number of input features.
        param out_features (int): Number of output features.
        param lora_rank (int, optional): Rank of LoRA decomposition. Default is 4.
        param lora_scaler (float, optional): Scaler for LoRA weights. Default is 32.0.
        param use_dora (bool, optional): Whether to use DoRA (Weight-Decomposed Low-Rank Adaptation). Default is False.
        param quant (bool, optional): Whether to apply weight quantization. Default is False.
        param plora_steps (Union(int, None), optional): Steps to merge and reset lora weight.  Deault is None.
        """
        super().__init__(in_features, out_features, bias=False)
        self.lora_rank = lora_rank
        self.lora_scaler = lora_scaler / lora_rank
        self.quant = quant
        self.dora = use_dora
        self.plora = plora_steps is not None

        self._init_lora_weights()
        # Enable plora, if plora_steps is not None.
        if self.plora:
            self.plora_steps = plora_steps
            self.plora_counter = 0
    

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

            # Weather to use DoRA.
            if self.dora and lora_weight is not None:
                weight = self._apply_dora(weight, lora_weight)
            elif lora_weight is not None:
                weight = weight + lora_weight
        # Unified output.
        return F.linear(x, weight)

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
        if self.quant:
            self.weight_a = nn.Parameter(
                torch.empty((self.lora_rank, self.in_features), dtype=torch.int8, requires_grad=False)
            )
            self.weight_b = nn.Parameter(
                torch.zeros((self.out_features, self.lora_rank), dtype=torch.int8, requires_grad=False)
            )
            self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
            self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.weight_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features)))
            self.weight_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank)))
        std = (1 / self.in_features) ** 0.5
        nn.init.normal_(self.weight_a, mean=0, std=std)

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
        return hasattr(self, 'weight_a') and hasattr(self, 'weight_b')

    def _compute_lora(self): 
        if self.has_lora_weights:
            # Compute lora weight.
            weight_a = self._quantize_weight(self.weight_a, self.weight_a_quantizer)
            weight_b = self._quantize_weight(self.weight_b, self.weight_b_quantizer)
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

    def merge_and_reset(self):
        # If there is lora weight and it has been successfully merged, reinitialize the lora weight:
        if self._merge_lora():
            std = (1 / self.in_features)**0.5
            nn.init.normal_(self.weight_a, mean=0, std=std)
            nn.init.zeros_(self.weight_b)
            if self.quant:
                self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
                self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

    def merge_and_del(self):
        # If there is lora weight and it has been successfully merged, delete all lora attrs:
        if self._merge_lora():
            delattr(self, "weight_a")
            delattr(self, "weight_b")
            if self.quant:
                delattr(self, "weight_a_scaler")
                delattr(self, "weight_b_scaler")

    def reset(self):
        if not self.has_lora_weights:
            self._init_lora_weights()

    def print_details(self) -> None:
        print(f"LinearWithLoRA Layer: in_features={self.in_features}, out_features={self.out_features}")
        print(f"Lora Enabled: {self.has_lora_weights}, LoRA Rank: {self.lora_rank}, Quantized: {self.quant}, DoRA: {self.dora}")


def switch_to_lora(model, replace_names, rank=4, lora_scaler=32, transposition=False, use_dora=False, plora_steps=None):
    """
    Switch function for lora, responsible for replacing Linear layer with LinearWithLoRA layer

    param model: Any pytorch model.
    param replace_names: List of module names to be replaced by LoRA.
    param rank: Rank for LoRA.
    param lora_scaler: Scaler for LoRA.
    """
    if replace_names is None:
        replace_names = ['qkv_proj']
    for name, module in model.named_modules():
        for replace_name in replace_names:
            if isinstance(module, nn.Module) and replace_name in name:
                # Create LoRA layer instance.
                assert all(hasattr(module, attr) for attr in ["in_features", "out_features", "weight"]), \
                "Module is missing one or more of the required attributes: 'in_features', 'out_features', 'weight'"

                quant = getattr(module, "quant", False)
                lora_layer = LinearWithLoRA(lora_rank=rank, 
                                            lora_scaler=lora_scaler, 
                                            in_features=module.in_features, 
                                            out_features=module.out_features, 
                                            use_dora=use_dora, 
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

def get_parent_model(parent_model, module):
    """
    Find the parent module for the input module recursively.

    param parent_model: Root model for the search.
    param module: Submodule to find the parent module for.

    Returns:
    Parent module if found, None otherwise.
    """
    for _, sub_module in parent_model._modules.items():
        if sub_module is module:
            return parent_model
        parent = get_parent_model(sub_module, module)
        if parent:
            return parent
    return None

if __name__ == '__main__':
    use_dora = False
    plora_steps = None
    # initialize test
    linear = LinearWithLoRA(2048, 2048, 8, 32, use_dora, False, plora_steps)
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
    switch_to_lora(model, ['norm'], transposition=True) # result in a assert error 

    import torch.optim as optim
    # backward test
    class TestModel(nn.Module):
        def __init__(self, in_features, out_features, lora_rank, lora_scaler, use_dora, quant, plora_steps):
            super().__init__()
            self.linear = LinearWithLoRA(in_features, out_features, lora_rank, lora_scaler, use_dora, quant, plora_steps)

        def forward(self, x):
            return self.linear(x)

    def test_lora_gradient():
        # Set up the model
        in_features = 64
        out_features = 64
        lora_rank = 4
        lora_scaler = 32.0
        use_dora = False
        quant = False
        plora_steps = None

        model = TestModel(in_features, out_features, lora_rank, lora_scaler, use_dora, quant, plora_steps)
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
        for param in model.parameters():
            assert param.grad is not None, "Gradient is None for some parameters"

        print("Test passed: Gradients are not None.")

    test_lora_gradient()