from common.lora_modules.lora import *
from common.lora_modules.adalomo import AdaLomo
from common.utils import print_rank_0, to_device, Timer

class LinearWithLoRADA(LinearWithLoRA):
    def delta_w_reinit(self,
                        global_rank: int = 0,
                        is_first: bool = False):
        if not hasattr(self.weight, "cpu_weight"):
            print("No cpu_weight found. Skipping LoRA-DA init.")
            return
        
        delta_w = self.weight.data.float().cuda() - self.weight.cpu_weight.float().cuda()
        
        U, S, V = torch.svd_lowrank(delta_w, q=self.lora_rank, niter=4)
        V = V.T
        B = U @ (torch.diag(torch.sqrt(S)))
        A = (torch.diag(torch.sqrt(S)) @ V)
        
        self.weight_a.data = A.contiguous().to(self.weight.device)
        self.weight_b.data = B.contiguous().to(self.weight.device)
        
        if hasattr(self.weight, "cpu_weight"):
            self.weight.data.copy_(self.weight.cpu_weight.to(self.weight.device))
            del self.weight.cpu_weight

        if is_first:
            print_rank_0(f'--->LoRA-DA re-init example: weight_B->{self.weight_b}', global_rank)


def lora_da_reinit(model, dataloader, args, iters=1):
    print_rank_0("--->Estimating gradient for lora da.", rank=args.global_rank)
    torch.cuda.empty_cache()
    
    with Timer() as timer:
        model.to(args.device)
        model.train()

        for module in model.modules():
            # Disable requires_grad for all parameters in the module
            for param in module.parameters():
                param.requires_grad = False
            # Enable requires_grad only for the weight parameter of LinearWithLoRADA
            if isinstance(module, LinearWithLoRADA):
                module.weight.requires_grad = True
                module.weight.cpu_weight = module.weight.data.detach().clone().cpu()

        optimizer = AdaLomo(
            model,
            lr=args.lora_da_lr
        )

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            loss = model(**batch)[0]
            optimizer.fused_backward(loss, args.lora_da_lr)
            
            print_rank_0(f'--->LoRA-DA gradient computing step: {idx+1}, loss: {loss.item()}', args.global_rank)
            if (idx + 1) == iters:
                break
            
        is_first = True
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRADA):
                print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
                module.delta_w_reinit(
                    global_rank=args.global_rank, 
                    is_first=is_first
                )
                is_first = False

        for hook in optimizer.hooks:
            hook.remove()

        del optimizer

    print_rank_0(f'--->Total time consumed: {timer.time_cost}', args.global_rank)