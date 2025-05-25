""" Implements MoLA"""

from common.lora_modules.lora_moe import *

class LinearWithMoLA(LinearWithLoRAMoE):
    def init_lora_weights(self):
        pass

    def init_mola_weights(self):
        super().init_lora_weights()

def init_mola_experts_by_shape(model, args):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and 'layer' in name:
            num_layers = len(module)
            quarter = num_layers // 4
            half = num_layers // 2
            three_quarters = 3 * quarter

            for layer_idx, layer in enumerate(module):
                for layer_module in layer.modules():
                    if isinstance(layer_module, LinearWithMoLA):
                        if layer_idx < quarter:
                            stage = 'first_quarter'
                        elif layer_idx < half:
                            stage = 'second_quarter'
                        elif layer_idx < three_quarters:
                            stage = 'third_quarter'
                        else:
                            stage = 'last_quarter'

                        n_experts = args.lora_moe_n_experts
                        if args.mola_type == 'triangle':
                            if stage == 'first_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 4)
                            elif stage == 'second_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 2)
                            elif stage == 'third_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 0.5)
                            elif stage == 'last_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 0.25)

                        elif args.mola_type == 'invert_triangle':
                            if stage == 'first_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 0.25)
                            elif stage == 'second_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 0.5)
                            elif stage == 'third_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 2)
                            elif stage == 'last_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 4)

                        elif args.mola_type == 'hourglass':
                            if stage == 'first_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 2)
                            elif stage == 'last_quarter':
                                layer_module.lora_moe_n_experts = int(n_experts * 2)


                        elif args.mola_type == 'rectangle':
                            pass

                        else:
                            raise ValueError(f'MoLA type must in ["triangle", "invert_triangle", "hourglass", "rectangle"], got {args.mola_type}')
                        
                        layer_module.init_mola_weights()