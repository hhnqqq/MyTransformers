import math
import random
from collections import OrderedDict
from torch import Tensor, svd_lowrank as fast_svd
from torch.linalg import svd as standard_svd

from common.lora_modules.lora import *
from common.utils.utils import Timer, reduce_tensor, to_device, print_rank_0

class LinearWithTDLoRA(LinearWithLoRA):
    def __init__(
        self,
        lora_config: LoRAConfig,
        fast_svd_n_iters: Optional[int] = 16,
        tdlora_init_method: str = 'weight_svd'
    ):
        self.n_iters = fast_svd_n_iters
        self.fast_svd = fast_svd_n_iters > 2
        self.init_method = tdlora_init_method
        assert tdlora_init_method in ['vanilla', 'weight_svd', 'grad_svd']
        super().__init__(lora_config)

    def init_lora_weights(self):
        self.lora_rank = 0

    def _lora_forward(self, x: Tensor, result: Tensor) -> Tensor:
        if self.lora_rank == 0:
            return result
        else:
            return super()._lora_forward(x, result)
        
    def dynamic_init(self, rank):
        self.lora_rank = rank
        if self.init_method == 'weight_svd':
            self.weight_svd_init()
        elif self.init_method == 'grad_svd':
            self.grad_svd_init()
        elif self.init_method == 'vanilla':
            super().init_lora_weights()
        del self.weight.grad_stored

    def grad_svd_init(self, 
                     direction: str = 'ArB2r', 
                     scale: str = 'stable', 
                     stable_gamma: int = 16, 
                     scaling_factor: int = 16):
        if not hasattr(self.weight, 'grad_stored'):
            return
        
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
        elif direction == "random":
            random_list = random.sample(range(2 * self.lora_rank), 2 * self.lora_rank)
            indexes_A = random_list[0:self.lora_rank]
            indexes_B = random_list[self.lora_rank:2 * self.lora_rank]
            B = U[:, indexes_B]
            A = V[indexes_A, :]
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
            _, S, _ = fast_svd(self.weight.float(), q=4 * self.lora_rank, niter=4)
            S /= scaling_factor
            avg_s = torch.sqrt(S[:self.lora_rank]).mean().to(A.device)
            A *= avg_s
            B *= avg_s
        elif scale != "unit":
            raise ValueError(f"Unknown scale: {scale}")

        # Update the LoRA weights
        self.weight_a = nn.Parameter(A.contiguous().cuda())
        self.weight_b = nn.Parameter(B.contiguous().cuda())
        
        weight_dtype = self.weight.dtype
        weight = self.weight.to(torch.float32)
        self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

    def weight_svd_init(self):
        if self.lora_rank > 0:
            dtype = self._get_lora_dtype()
            weight_dtype = self.weight.dtype
            requires_grad = not self.quant

            weight = self.weight.to(torch.float32)
            if self.fast_svd:
                # Run fast svd.
                Vr, Sr, Ur = fast_svd(weight.data, self.lora_rank, niter=self.n_iters)
                Uhr = Ur.t()
            else:
                # Full svd, which is very slow.
                V, S, Uh = standard_svd(self.weight.data, full_matrices=False)
                Vr, Sr, Uhr = V[:, :self.lora_rank], S[:self.lora_rank], Uh[:self.lora_rank]

            Sr.div_(self.lora_scaler) 
            sqrt_Sr = Sr.sqrt_()
            
            weight_a_data = torch.diag(sqrt_Sr) @ Uhr
            self.weight_a = nn.Parameter(weight_a_data.to(dtype), requires_grad=requires_grad)
            weight_b_data = Vr @ torch.diag(sqrt_Sr)
            self.weight_b = nn.Parameter(weight_b_data.to(dtype), requires_grad=requires_grad)

            if self.quant:
                self.weight_a_scaler = nn.Parameter(torch.Tensor(self.lora_rank))
                self.weight_b_scaler = nn.Parameter(torch.Tensor(self.out_features))

            self.weight.data = (weight - self._compute_lora_weight()).to(weight_dtype)

def get_record_gradient_hook(model):
    def record_gradient_hook(grad):
        torch.cuda.synchronize()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                if not hasattr(p, 'grad_stored'):
                    p.grad_stored = p.grad.cpu()
                else:
                    p.grad_stored += p.grad.cpu()
                p.grad = None
        return grad

    return record_gradient_hook

def get_allocated_rank(model, 
                       avg_lora_rank, 
                       global_rank, 
                       max_lora_rank=9999, 
                       min_lora_rank=1, 
                       temperature=0.5):
    named_ranks = {}
    named_importances = OrderedDict()
    total_budget = 0
    named_features = {}

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, LinearWithTDLoRA):
                # Calculate the importance
                if not hasattr(module.weight, 'grad_stored'):
                    print_rank_0(f'--->Module: {name} do not have stored gradients', global_rank)
                    continue
                importance = torch.mean(torch.linalg.norm((torch.abs(module.weight.data * module.weight.grad_stored.cuda())), dim=1))
                # importance = torch.log1p(importance) 对数缩放，防止一边倒
                named_importances[name] = importance

                # Calculate features and budget
                features = module.in_features + module.out_features
                named_features[name] = features
                total_budget += features * avg_lora_rank

        if not named_importances:
            raise ValueError("No gradients were stored. Check if backward pass was performed correctly.")

        # Calculate softmax of importances
        importances_tensor = torch.tensor(list(named_importances.values()))
        importances_tensor = (importances_tensor - importances_tensor.min()) / (importances_tensor.max() - importances_tensor.min())
        softmaxed_importances = torch.softmax(importances_tensor / temperature, dim=0)
        print_rank_0(f'TDLoRA softmaxed importance: {softmaxed_importances}', global_rank)

        # Allocate ranks based on calculated budgets
        for name, softmaxed_importance in zip(named_importances.keys(), softmaxed_importances):
            new_budget = math.ceil(total_budget * softmaxed_importance.item())
            rank = new_budget // named_features[name]
            named_ranks[name] = min(max(math.ceil(rank), min_lora_rank), max_lora_rank)

    return total_budget, named_ranks

def tdlora_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    iters: int = 1,
):
    print_rank_0("--->Estimating gradient for tdlora.", rank=args.global_rank)
    with Timer() as timer:
        model.to(args.device)
        model.train()

        # Note that we only compute gradient for LoRA-GA layers.
        # Avoiding unnecessary computing.
        hooks = [
            module.weight.register_hook(get_record_gradient_hook(model))
            for module in model.modules()
            if isinstance(module, LinearWithTDLoRA)
        ]

        for module in model.modules():
            if isinstance(module, LinearWithTDLoRA):
                module.weight.requires_grad = True

        for idx, batch in enumerate(dataloader):
            batch = to_device(batch, args.device)
            output = model(**batch)
            loss = output.loss if args.huggingface else output[0]
            loss.backward()
            print_rank_0(f'--->TDLoRA gradient computing step: {idx+1}, loss: {loss.item()}, remaining steps: {iters - (idx+1)} ', args.global_rank)

            # for p in model.parameters():
            #     p.grad = None

            if (idx + 1) == iters:
                break

        for hook in hooks:
            hook.remove()

        torch.cuda.empty_cache()
        torch.distributed.barrier()

        print_rank_0('--->All reduce TDLoRA stored gradients if needed.', args.global_rank)
        for p in model.parameters():
            if hasattr(p, 'grad_stored'):
                p.grad_stored /= iters
                if args.world_size > 1:
                    p.grad_stored = reduce_tensor(p.grad_stored.to(args.device), args.world_size).to('cpu')

        total_budget, named_ranks = get_allocated_rank(model, args.lora_rank, args.global_rank)
        print_rank_0(f'--->TDLoRA total budget: {total_budget}', args.global_rank)
        for name, module in model.named_modules():
            if isinstance(module, LinearWithTDLoRA) and name in named_ranks.keys():
                print_rank_0(f'--->Module {name} is initiating lora weight, rank is: {named_ranks[name]}', args.global_rank)
                module.dynamic_init(named_ranks[name])

    print_rank_0(f'--->Total time consumed for TDLoRA initialization: {timer.time_cost}', args.global_rank)