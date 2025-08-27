
from common.lora_modules.ralora import *
from common.lora_modules.adalomo import AdaLomo

def compute_importance(param, grad_stored):
    param = param.float()
    grad_stored = grad_stored.float().to(param.device)
    importance = torch.linalg.matrix_norm(grad_stored).item()
    return importance
    
def get_allocated_rank(model, args):
    named_ranks = {}
    named_importances = OrderedDict()
    total_budget, smooth_total_budget, actual_trainable = 0, 0, 0
    named_features, named_smooth_features = {}, {}

    feature_adjust_func: Callable = {
        None: lambda x: x
    }.get(args.ralora_features_func, lambda x: x)

    if args.global_rank == 0:
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, LinearWithRaLoRA):
                    if not hasattr(module.weight, 'cpu_weight'):
                        print_rank_0(f'--->Module: {name} do not have cpu_weight', args.global_rank)
                        continue
                    features = module.in_features + module.out_features
                    # Move gradient to GPU one at a time
                    pretrained_weight = module.weight.cpu_weight.cuda()
                    delta_w = module.weight - pretrained_weight
                    importance = compute_importance(
                        pretrained_weight,
                        delta_w
                    )
                    named_importances[name] = importance
                    adjusted_features = feature_adjust_func(features)
                    named_smooth_features[name] = adjusted_features
                    named_features[name] = features
                    smooth_total_budget += adjusted_features * args.lora_rank
                    total_budget += features * args.lora_rank
                    # Clear GPU gradient after use
                    delta_w = None
                    pretrained_weight = None
                    torch.cuda.empty_cache()

            if not named_importances:
                raise ValueError("No cpu_weight were stored. Check if backward pass was performed correctly.")

            importances_tensor = torch.tensor(list(named_importances.values()))
            normalized_importances = get_normalized_importances(importances_tensor)

            for name, normalized_importance in zip(named_importances.keys(), normalized_importances):
                smooth_trainable = round(smooth_total_budget * normalized_importance.item())
                rank = smooth_trainable // named_smooth_features[name]
                if args.ralora_max_rank and args.ralora_min_rank:
                    named_ranks[name] = min(max(round(rank), args.ralora_min_rank), args.ralora_max_rank)
                else:
                    named_ranks[name] = rank
                actual_trainable += named_ranks[name] * named_features[name]

    else:
        total_budget, actual_trainable, named_importances = 0, 0, OrderedDict()
        named_ranks = {}
    # Broadcast named_ranks and has_converged to all ranks
        
    if args.world_size > 1:
        dist.barrier()
        
        if args.global_rank == 0:
            broadcast_data = {
                'named_ranks': named_ranks
            }
            serialized = pickle.dumps(broadcast_data)
            data = torch.ByteTensor(list(serialized)).to(args.device)
            length = torch.tensor([len(serialized)], dtype=torch.long, device=args.device)
        else:
            length = torch.tensor([0], dtype=torch.long, device=args.device)
            data = torch.empty(1024*1024, dtype=torch.uint8, device=args.device)  # Allocate sufficient space
            
        dist.broadcast(length, src=0)
        
        if args.global_rank != 0:
            data = torch.empty(length.item(), dtype=torch.uint8, device=args.device)
        dist.broadcast(data, src=0)
        
        if args.global_rank != 0:
            serialized = bytes(data.cpu().tolist())
            broadcast_data = pickle.loads(serialized)
            named_ranks = broadcast_data['named_ranks']
    return total_budget, actual_trainable, named_ranks, named_importances

def compute_n_split_allocations(model, named_ranks, args):
    """
    Compute the optimal number of mini LoRA modules for each layer
    based on gradient importance.
    
    Returns a dictionary mapping module names to the number of 
    mini LoRA modules to use for that layer.
    """
    named_n_splits = {}
    named_eranks = {}
    print_rank_0(f'--->Allocate n using the erank method.', args.global_rank)
    print_rank_0(f'--->The max power is {args.ralora_erank_max_power}.', args.global_rank)
    min_power = 0
    max_power = args.ralora_erank_max_power
    if args.global_rank == 0:
        start_time = time.time()
        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA):
                if not hasattr(module.weight, 'cpu_weight'):
                        print_rank_0(f'--->Module: {name} does not have cpu_weight', args.global_rank)
                        continue
                delta_w = module.weight - module.weight.cpu_weight.cuda()
                if args.ralora_svd_threshold > 0:
                    # Count the number of singular values above the threshold
                    print_rank_0(f'--->Module {name} is calculating erank using Threshold svd', args.global_rank)
                    erank = count_singular_values_above_threshold(delta_w, 
                                                                  threshold=args.ralora_svd_threshold, 
                                                                  dtype=torch.float32)
                elif args.ralora_cumulative_variance_threshold > 0:
                        # Count the number of singular values that contribute to the cumulative variance
                        print_rank_0(f'--->Module {name} is calculating erank using cumulative variance svd', args.global_rank)
                        erank = count_singular_values_by_variance_threshold(delta_w, 
                                                                            cumulative_variance_threshold=args.ralora_cumulative_variance_threshold, 
                                                                            dtype=torch.float32)
                else:
                    erank = compute_effective_rank(delta_w)
                named_eranks[name] = erank
                delta_w = None
        end_time = time.time()
        print_rank_0(f'--->Time consumption for calculating svd: {end_time-start_time:.6f}s', args.global_rank)
        if not named_eranks:
                print_rank_0(f'--->No gradient erank calculated for dynamic n allocation', args.global_rank)
    
        if args.ralora_allocate_by_erank:
            result = named_eranks
        else:
            # Allocating n according erank and lora rank
            assert args.ralora_erank_max_power is not None, f'The eran_max_power must be setted.'
            for name, erank in named_eranks.items():
                n_splits_power = min(max_power, max(min_power, math.floor(math.log2(erank) - math.log2(named_ranks[name]))))
                named_n_splits[name] = 2 ** n_splits_power
                print_rank_0(f'--->Module {name}: grad_erank={math.ceil(erank)},  n_split={named_n_splits[name]}', args.global_rank)
            result = named_n_splits
    if args.world_size > 1:
        dist.barrier()
        
        if args.global_rank == 0:
            broadcast_data = {
                'result': result
            }
            serialized = pickle.dumps(broadcast_data)
            data = torch.ByteTensor(list(serialized)).to(args.device)
            length = torch.tensor([len(serialized)], dtype=torch.long, device=args.device)
        else:
            length = torch.tensor([0], dtype=torch.long, device=args.device)
            data = torch.empty(1024*1024, dtype=torch.uint8, device=args.device)  # Allocate sufficient space
            
        dist.broadcast(length, src=0)
        
        if args.global_rank != 0:
            data = torch.empty(length.item(), dtype=torch.uint8, device=args.device)
        dist.broadcast(data, src=0)
        
        if args.global_rank != 0:
            serialized = bytes(data.cpu().tolist())
            broadcast_data = pickle.loads(serialized)
            result = broadcast_data['result']
    return result

def dralora_reinit(model, dataloader, args, iters=1):
    print_rank_0("--->Estimating gradient for dralora.", rank=args.global_rank)
    torch.cuda.empty_cache()
    
    with Timer() as timer:
        model.to(args.device)
        model.train()

        for module in model.modules():
            # Disable requires_grad for all parameters in the module
            for param in module.parameters():
                param.requires_grad = False
            # Enable requires_grad only for the weight parameter of LinearWithLoRADA
            if isinstance(module, LinearWithRaLoRA):
                module.weight.requires_grad = True
                if args.global_rank == 0:
                    # offload the cpu_weight by rank-0 process only.
                    module.weight.cpu_weight = module.weight.data.detach().clone().cpu()
        optimizer = AdaLomo(
            model,
            lr=args.adalomo_lr
        )
        optimizer.cal_importance_step = 5

        for idx, batch in enumerate(dataloader):
            timer.average_time("start")
            batch = to_device(batch, args.device)
            loss = model(**batch)[0]
            optimizer.fused_backward(loss, args.adalomo_lr)
            timer.average_time("end")

            print_rank_0(f'--->DRaLoRA gradient computing step: {idx+1}, loss: {loss.item()}, time_cost: {timer.loop_time:.2f}s, peak_memory: {timer.peak_memory}MB', args.global_rank)
            if (idx + 1) == iters:
                break

        for hook in optimizer.hooks:
            hook.remove()

        erank_dict = optimizer.erank_dict
        importance_dict = optimizer.importance_dict
        
        del optimizer

        if args.world_size > 1:
            torch.distributed.barrier()

        named_n_splits, named_importances = {}, {}
        if args.ralora_allocate_by_erank:
            named_ranks = compute_n_split_allocations(model, {}, args)
        else:
            total_budget, actual_trainable, named_ranks, named_importances = get_allocated_rank(model, args)
            
            # Compute and allocate optimal number of mini LoRA modules
            if not args.ralora_disable_n_split:
                print_rank_0('--->Computing dynamic n allocation for Mini-LoRA blocks', args.global_rank)
                named_n_splits = compute_n_split_allocations(model, named_ranks, args)

            print_rank_0(f'--->DRaLoRA total budget: {total_budget}, actual trainable: {actual_trainable}', args.global_rank)

        save_floder = os.path.join(args.output_path, args.experiment_name)

        named_ranks = {k:math.ceil(v) for k, v in named_ranks.items()}
        ensure_directory_exists(save_floder, args.global_rank)
        if args.global_rank == 0:
            with open(os.path.join(save_floder, 'rank.json'), 'w') as f:
                json.dump(named_ranks, f)
            with open(os.path.join(save_floder, 'importance.json'), 'w') as f:
                json.dump(named_importances, f)
            if erank_dict:
                with open(os.path.join(save_floder, 'erank.json'), 'w') as f:
                    json.dump(erank_dict, f)
            if importance_dict:
                with open(os.path.join(save_floder, 'step_importance.json'), 'w') as f:
                    json.dump(importance_dict, f)
            if named_n_splits:
                with open(os.path.join(save_floder, 'n_splits.json'), 'w') as f:
                    json.dump({name: int(n) for name, n in named_n_splits.items()}, f)

        for name, module in model.named_modules():
            if isinstance(module, LinearWithRaLoRA) and name in named_ranks.keys():
                n_split = named_n_splits.get(name, 1)
                torch.distributed.barrier()
                cpu_weight = module.weight.cpu_weight.to(args.device) if args.global_rank == 0 else torch.zeros_like(module.weight).to(args.device)
                # boardcast the cpu_weight to other ranks.
                torch.distributed.broadcast(cpu_weight, src=0)
                print_rank_0(f'--->Module {name} is initiating lora weight, rank: {named_ranks[name]}, n_split: {n_split}', args.global_rank)
                module.dynamic_init(args.lora_rank, named_ranks[name], n_split=n_split)
                module.weight.data.copy_(cpu_weight)
                if hasattr(module.weight, "cpu_weight"):
                    del module.weight.cpu_weight
            
        torch.cuda.empty_cache()

    print_rank_0(f'--->Total time consumed for RaLoRA initialization: {timer.time_cost}', args.global_rank)