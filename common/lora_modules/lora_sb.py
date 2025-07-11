from common.lora_modules.mos_lora import *
from common.lora_modules.lora_ga import *
from common.utils.utils import print_rank_0


class LinearWithLoRASB(LinearWithMosLoRA):
    def gradient_reinit(self,
                        global_rank: int = 0,
                        is_first: bool = False,
                        reset_weight: bool = False):
        if not hasattr(self.weight, 'grad_stored'):
            print('no gradients')
            return
        
        # Perform SVD on the weight gradient
        # Weight stored gradient shape [out_feature, in_feature]
        B, S, A = torch.svd_lowrank(self.weight.grad_stored.float().cuda(), q=self.lora_rank, niter=4)
        # U shape [out_feature, r] S shape [r, r] V shape [in_feature, r]
        A = A.T
        
        del self.weight.grad_stored

        # Update the LoRA weights
        self.weight_a.data = A.contiguous().cuda()
        self.weight_b.data = B.contiguous().cuda()
        self.weight_ab_mixer.data = torch.diag(S).contiguous().cuda()
        
        if reset_weight:
            self.reset_weight()

        if is_first:
            print_rank_0(f'--->LoRA-GA re-init example: weight_B->{self.weight_b}', global_rank)

    def reset_weight(self):
        weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

def lora_sb_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
):
    print_rank_0("--->Estimating gradient for lora sb.", rank=args.global_rank)
    torch.cuda.empty_cache()
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for LoRA-SB layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model))
            for module in model.modules()
            if isinstance(module, LinearWithLoRASB)
        ]

        for module in model.modules():
            if isinstance(module, LinearWithLoRASB):
                module.weight.requires_grad = True
            elif isinstance(module, torch.nn.Linear):
                module.weight.requires_grad = False

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            loss = model(**batch)[0]
            loss.backward()
            print_rank_0(f'--->LoRA-SB gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)} ', args.global_rank)

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        torch.cuda.empty_cache()
        if args.world_size > 1:
            torch.distributed.barrier()

        print_rank_0('--->All reduce LoRA-SB stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):
                p.grad_stored /= iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to('cpu')

        is_first = True
        for name, module in model.named_modules():
            if isinstance(module, LinearWithLoRASB):
                print_rank_0(f'--->Module {name} is reinitiating lora weight', args.global_rank)
                module.gradient_reinit(global_rank=args.global_rank, 
                                       is_first=is_first,
                                       reset_weight=args.lora_reset_weight)
                is_first = False


    print_rank_0(f'--->Total time consumed for LoRA-SB initialization: {timer.time_cost}', args.global_rank)