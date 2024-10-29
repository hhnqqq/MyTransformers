import time
import torch
import torch.nn as nn
from common.lora_modules.lorapro_optim import *
from common.lora_modules import *

# initialize test
test_class = LinearWithLoRA
config = LoRAConfig(in_features=2048, 
                    out_features=2048, 
                    lora_rank=8, 
                    lora_scaler=32, 
                    quant=False)
linear = test_class(config)
linear.weight.data = torch.randn(2048,2048)

print(linear.weight)
linear.merge_and_reset()
print(linear.weight)

# forward test
print(linear(torch.randn(2048,2048)))
linear.print_details()

# backward test
class TestModel(nn.Module):
    def __init__(self, in_features, out_features, lora_rank, lora_scaler, lora_dropout, quant):
        super().__init__()
        config = LoRAConfig(in_features, out_features, lora_rank, lora_scaler, lora_dropout, quant)
        self.linear = test_class(config)

    def forward(self, x):
        return self.linear(x)

def test_lora_gradient():
    # Set up the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_features = 4096
    out_features = 4096
    lora_rank = 128
    lora_scaler = 32.0
    lora_dropout = 0.1
    quant = False

    model = TestModel(in_features, out_features, lora_rank, lora_scaler, lora_dropout, quant)
    model.to(device)
    # model.linear.merge_and_del()

    # Generate some random input and target
    input_data = torch.randn(32, in_features).to(device)
    target_data = torch.randn(32, out_features).to(device)
    
    # Initialize the Adam optimizer
    optimizer = LoRAProAdamW({'params':model.named_parameters()}, lr=0.001)

    # Forward pass
    start = time.time()
    output = model(input_data)
    forward_end = time.time()
    
    # Compute the loss
    loss = nn.MSELoss()(output, target_data)

    # Backward pass
    optimizer.zero_grad()  # Zero the gradients
    loss.backward()
    optimizer.step()  # Update model parameters
    end = time.time()

    # Check if the gradients are not None
    print(f'forward path spend time {forward_end-start:.6f}s')
    print(f'total spend time {end-start:.6f}s')
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"{name}'s gradient is None")

    print("Test passed: Gradients are not None.")

test_lora_gradient()

"""
in_features = 4096
out_features = 4096
rank:128

lora:forward path spend time 0.050580s
total spend time 0.082057s

melora:forward path spend time 0.059647s
total spend time 0.100974s(split=2)
forward path spend time 0.063676s
total spend time 0.099424(split=16)
forward path spend time 0.100062s
total spend time 0.148700s(split=128)

moslora:forward path spend time 0.049974s
total spend time 0.082502s

dora:forward path spend time 0.068298s
total spend time 0.848875s
"""