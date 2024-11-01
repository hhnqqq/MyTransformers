from collections.abc import Iterable

from common.utils import print_rank_0
from common.utils import to_device
from common.lora_modules.lora import *

class LinearWithLoRAGA(LinearWithLoRA):
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
    model.to(args.device)
    model.train()
    hooks = []
    for param in model.parameters():
        param.requires_grad = True
        hook = param.register_hook(get_record_gradient_hook(model))
        hooks.append(hook)
    for idx, batch in enumerate(dataloader):
        batch = to_device(batch, args.device)
        output = model(**batch)
        if args.huggingface:
            output.loss.backward()
        else:
            output[0].backward()
        get_record_gradient_hook(model)(None)
        for p in model.parameters():
            if p.grad is not None:
                p.grad = None
        if (idx+1) == iters:
            break
    for p in model.parameters():
        p.grad_stored /= iters
    for hook in hooks:
        hook.remove()
    torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRAGA):
            print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
            module.gradient_reinit()