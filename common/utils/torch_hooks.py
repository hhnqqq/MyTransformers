import torch
import common.utils.parallel_states as parallel_state

import numpy as np


def save_grad(name):
    def hook(grad):
        # grads[name] = grad
        if parallel_state.get_data_parallel_rank() == 0:
            print(f"Tensor name={name}, Gradient={grad}")
    return hook

def hook_forward_fn(module, input, output):
    if parallel_state.get_data_parallel_rank() == 0:
        print("It's forward: ")
        print(f"Module: {module}")
        print(f"Input tensor: {input}")
        print(f"Output tensor: {output}")
        print("="*20)
        
def hook_backward_norm_fn(module, grad_input, grad_output):
    if parallel_state.get_data_parallel_rank() == 0:
        print("It's backward:")
        print(f"Module: {module}")
        
        if grad_input is not None and len(grad_input) > 0:
            input_grad_norm = torch.norm(torch.stack([torch.norm(gi) for gi in grad_input if gi is not None]), 2)
            print(f"Input gradient 2-order norm: {input_grad_norm}")
        
        if grad_output is not None and len(grad_output) > 0:
            output_grad_norm = torch.norm(torch.stack([torch.norm(go) for go in grad_output if go is not None]), 2)
            print(f"Output gradient 2-order norm: {output_grad_norm}")
        
        param_grad_norm = torch.norm(
            torch.stack([torch.norm(param.grad) for name, param in module.named_parameters() if param.grad is not None]), 
            2
        ).item()
        print(f"Module parameters gradient 2-order norm: {param_grad_norm}")
        
        print("="*20)

def hook_backward_fn(module, grad_input, grad_output):
    if parallel_state.get_data_parallel_rank() == 0:
        print("It's backward: ")
        print(f"Module: {module}")
        print(f"Input gradient: {grad_input}")
        print(f"output graidnet: {grad_output}")
        print("="*20)

def hook_detect_anomaly(name):
    def hook(module, input, output):
        if parallel_state.get_data_parallel_rank() == 0:
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"Detected NaN or Inf in {name}")
    return hook

class ParameterUpdateHook:
    def __init__(self, param_name, param, print_every=100, update_norm_order=2):
        self.param_name = param_name
        self.param = param
        self.print_every = print_every
        self.step_count = 0
        self.prev_data = param.data.clone()
        self.update_norm_order = update_norm_order

    def __call__(self, grad):
        if parallel_state.get_data_parallel_rank() == 0:
            self.step_count += 1
            if self.step_count % self.print_every == 0:
                with torch.no_grad():
                    update = self.param.data - self.prev_data
                    update_norm = torch.norm(update, self.update_norm_order).item()  
                    print(f"Backward step {self.step_count}: {self.param_name} {self.update_norm_order} order update norm = {update_norm:.6f}")     
                    self.prev_data = self.param.data.clone()

class ModuleUpdateHook:
    def __init__(self, module, print_every=100, anomaly_threshold=5.0, update_norm_order=2):
        self.module = module
        self.print_every = print_every
        self.anomaly_threshold = anomaly_threshold
        self.step_count = 0
        self.param_states = {}
        self.update_norm_order = update_norm_order
        
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.param_states[name] = {
                    'prev_data': param.data.clone(),
                    'hook': param.register_hook(self.create_hook(name))
                }

    def create_hook(self, name):
        def hook(grad):
            if self.step_count % self.print_every == 0:
                with torch.no_grad():
                    param = self.module.get_parameter(name)
                    update = param.data - self.param_states[name]['prev_data']
                    update_norm = torch.norm(update).item()
                    
                    if 'update_norms' not in self.param_states[name]:
                        self.param_states[name]['update_norms'] = []
                    
                    self.param_states[name]['update_norms'].append(update_norm)
                    self.param_states[name]['prev_data'] = param.data.clone()

        return hook

    def step(self):
        self.step_count += 1
        
        if self.step_count % self.print_every == 0:
            all_update_norms = []
            anomalies = []

            for name, state in self.param_states.items():
                if 'update_norms' in state:
                    update_norm = state['update_norms'][-1]
                    all_update_norms.append(update_norm)
                    
                    if len(state['update_norms']) > 1:
                        mean = np.mean(state['update_norms'][:-1])
                        std = np.std(state['update_norms'][:-1])
                        if abs(update_norm - mean) > self.anomaly_threshold * std:
                            anomalies.append((name, update_norm, mean, std))

            avg_update_norm = np.mean(all_update_norms) if all_update_norms else 0

            print(f"Step {self.step_count}:")
            print(f"  Average update norm: {avg_update_norm:.6f}")
            
            if anomalies:
                print("  Anomalies detected:")
                for name, value, mean, std in anomalies:
                    print(f"    {name}: {value:.6f} (mean: {mean:.6f}, std: {std:.6f})")
            else:
                print("  No anomalies detected.")

    def close(self):
        for state in self.param_states.values():
            state['hook'].remove()