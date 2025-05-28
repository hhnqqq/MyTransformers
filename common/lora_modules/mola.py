""" Implements MoLA:Higher Layers Need More LoRA Experts (https://arxiv.org/pdf/2402.08562)"""

from enum import Enum
import torch
import torch.nn as nn
from common.lora_modules.lora_moe import LinearWithLoRAMoE

class MoLAType(Enum):
    """Supported MoLA architecture types"""
    TRIANGLE = "triangle"
    INVERT_TRIANGLE = "invert_triangle"
    HOURGLASS = "hourglass"
    RECTANGLE = "rectangle"

class LinearWithMoLA(LinearWithLoRAMoE):
    def init_lora_weights(self):
        pass

    def init_mola_weights(self):
        super().init_lora_weights()

class MoLAExpertManager:
    """
    Manages the distribution of experts across different layers in MoLA
    """
    @staticmethod
    def get_layer_stage(layer_idx: int, num_layers: int) -> str:
        """
        Determine the stage of a layer based on its index.

        Args:
            layer_idx: Index of the current layer
            num_layers: Total number of layers

        Returns:
            str: Stage identifier ('first_quarter', 'second_quarter', etc.)
        """
        quarter = num_layers // 4
        half = num_layers // 2
        three_quarters = 3 * quarter

        if layer_idx < quarter:
            return 'first_quarter'
        elif layer_idx < half:
            return 'second_quarter'
        elif layer_idx < three_quarters:
            return 'third_quarter'
        else:
            return 'last_quarter'

    @staticmethod
    def get_expert_multiplier(mola_type: MoLAType, stage: str) -> float:
        """
        Get the expert count multiplier based on MoLA type and layer stage.

        Args:
            mola_type: Type of MoLA architecture
            stage: Stage of the layer

        Returns:
            float: Multiplier for number of experts
        """
        multipliers = {
            MoLAType.TRIANGLE: {
                'first_quarter': 4.0,
                'second_quarter': 2.0,
                'third_quarter': 0.5,
                'last_quarter': 0.25
            },
            MoLAType.INVERT_TRIANGLE: {
                'first_quarter': 0.25,
                'second_quarter': 0.5,
                'third_quarter': 2.0,
                'last_quarter': 4.0
            },
            MoLAType.HOURGLASS: {
                'first_quarter': 4.0,
                'second_quarter': 0.25,
                'third_quarter': 0.25,
                'last_quarter': 4.0
            },
            MoLAType.RECTANGLE: {
                'first_quarter': 1.0,
                'second_quarter': 1.0,
                'third_quarter': 1.0,
                'last_quarter': 1.0
            }
        }
        return multipliers[mola_type][stage]

def init_mola_experts_by_shape(model: nn.Module, args) -> None:
    """
    Initialize MoLA experts across different layers of the model.

    Args:
        model: The neural network model
        args: Configuration arguments including mola_type and expert counts

    Raises:
        ValueError: If invalid MoLA type is specified
    """
    try:
        mola_type = MoLAType(args.mola_type.lower())
    except ValueError:
        raise ValueError(
            f'MoLA type must be one of {[t.value for t in MoLAType]}, '
            f'got {args.mola_type}'
        )

    for name, module in model.named_modules():
        # TODO: a MORE rigorous pattern.
        if isinstance(module, torch.nn.ModuleList) and 'layer' in name:
            num_layers = len(module)
            
            for layer_idx, layer in enumerate(module):
                for layer_module in layer.modules():
                    if isinstance(layer_module, LinearWithMoLA):
                        # Get stage and calculate number of experts
                        stage = MoLAExpertManager.get_layer_stage(layer_idx, num_layers)
                        multiplier = MoLAExpertManager.get_expert_multiplier(mola_type, stage)
                        n_experts = int(args.lora_moe_n_experts * multiplier)
                        
                        # Update module configuration
                        layer_module.lora_moe_n_experts = n_experts
                        
                        # Adjust top-k value based on number of experts
                        if n_experts == 1:
                            layer_module.moe_top_k = 1
                        elif layer_module.moe_top_k > n_experts:
                            layer_module.moe_top_k = 2
                        
                        # Initialize weights
                        layer_module.init_mola_weights()