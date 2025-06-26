"""Implementation of LoRA-One
Code reference: https://github.com/YuanheZ/LoRA-One/tree/main
"""

from common.utils import print_rank_0, reduce_tensor, Timer
from common.utils import to_device
from common.lora_modules.lora import *

class LinearWithLoRAOne(LinearWithLoRA):
    def gradient_reinit(self, 
                        stable_gamma: int = 128, 
                        global_rank: int = 0,
                        is_first: bool = False,
                        reset_weight: bool = False):
        if not hasattr(self.weight, 'grad_stored'):
            return
        
        # Perform SVD on the weight gradient
        # Weight stored gradient shape [out_feature, in_feature]
        U, S, V = torch.svd_lowrank(-self.weight.grad_stored.float().cuda(), q=4 * self.lora_rank, niter=4)
        # U shape [out_feature, 4r] S shape [4r, 4r] V shape [in_feature, 4r]
        V = V.T

        B = U[:, :self.lora_rank] @ torch.diag(torch.sqrt(S[:self.lora_rank])) / torch.sqrt(S[0])
        A = torch.diag(torch.sqrt(S[:self.lora_rank])) @ V[:self.lora_rank, :] / torch.sqrt(S[0])

        # Apply scaling to A and B based on the scale parameter
        B = B / stable_gamma**0.5
        A = A / stable_gamma**0.5

        # Update the LoRA weights
        self.weight_a.data = A.contiguous().cuda()
        self.weight_b.data = B.contiguous().cuda()
        
        if reset_weight:
            self.reset_weight()

        if is_first:
            print_rank_0(f'--->LoRA-One re-init example: weight_B->{self.weight_b}', global_rank)

    def reset_weight(self):
        weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

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

def lora_one_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
):
    print_rank_0("--->Estimating gradient for lora-One.", rank=args.global_rank)
    torch.cuda.empty_cache()
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for LoRA-One layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model))
            for module in model.modules()
            if isinstance(module, LinearWithLoRAOne)
        ]

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            if args.huggingface:
                loss = model(input_ids=batch['input_ids'],
                        labels=batch['labels'],
                        attention_mask=batch['attention_mask']).loss
            else:
                output = model(**batch)
                loss = output[0]
            print_rank_0(f'--->LoRA-One gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)} ', args.global_rank)

            for p in model.parameters():
                p.grad = None

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        torch.cuda.empty_cache()
        if args.world_size > 1:
            torch.distributed.barrier()

        print_rank_0('--->All reduce LoRA-One stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):
                p.grad_stored /= iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to('cpu')

        is_first = True
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRAOne):
                print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
                module.gradient_reinit(global_rank=args.global_rank, 
                                       is_first=is_first,
                                       reset_weight=args.lora_reset_weight)
                is_first = False


    print_rank_0(f'--->Total time consumed for LoRA-One initialization: {timer.time_cost}', args.global_rank)