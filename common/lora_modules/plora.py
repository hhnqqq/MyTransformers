# @author: haonan he
# @date: 2024-08-21
""" Implements PLORA"""

import torch.optim as optim

from typing import Optional
from common.lora_modules.lora import *

class LinearWithPLoRA(LinearWithLoRA):
    def __init__(self,
        lora_config:LoRAConfig,
        plora_steps: Optional[int] = None,
        optimizer: Optional[optim.Optimizer] = None):

        super().__init__(lora_config)
        self.plora_steps = plora_steps
        self.optimizer = optimizer
        self.plora_counter = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Every plora stage, we merge the origin lora weight and reset new lora weight.:
        self.plora_counter += 1
        if self.plora_counter == self.plora_steps:
            self.merge_and_reset()
            self.clear_optimizer_stat()
            self.plora_counter = 0

        return super().forward(x)
    
    def clear_optimizer_state(self):
        """
        Clear the optimizer state for the parameters of this module.
        """
        if self.optimizer is not None:
            for param in self.parameters():
                # Iterate over all parameter groups in the optimizer
                for group in self.optimizer.param_groups:
                    # Remove the state of the current module's parameters
                    if param in group['params']:
                        param_index = group['params'].index(param)
                        del group['params'][param_index]
                        self.optimizer.state.pop(param)