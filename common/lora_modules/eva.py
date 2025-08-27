# @author: jingqi ye (modified by haonan he)
"""
Implementation of EVA(Parameter Efficient Fine-tuning via Explained Variance Adaptation) [arxiv preprint]
Paper link: https://arxiv.org/abs/2410.07170
Code reference: https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py

EVA uses the directions capturing the most activation variance, 
provably maximizing the expected gradient signal and accelerating fine-tuning. 
EVA performs incremental SVD on minibatches of activation vectors and 
selects the right-singular vectors for initialization once they converged.
"""
import os
import json
import time
from typing import Callable
from torch import nn, Tensor

from typing import Union
from torch_incremental_pca import IncrementalPCA
from collections import Counter
from tqdm import tqdm
from functools import reduce

from common.lora_modules.lora import *
from common.utils.utils import Timer, to_device, print_rank_0, ensure_directory_exists

class LinearWithEVA(LinearWithLoRA):
    def __init__(self,
                lora_config: LoRAConfig):
        super().__init__(lora_config=lora_config)
        self.lora_alpha = lora_config.lora_scaler

    def prepare_init(self, lora_rank: int):
        self.lora_rank = lora_rank
        self.lora_scaler = self.lora_alpha / self.lora_rank  
    
    def init_lora_weights(self):
        pass

    def EVA_init(self, lora_rank: int):
        self.prepare_init(lora_rank)
        super().init_lora_weights()

    def EVA_reinit(self, lora_rank: int, weight_a: Tensor):
        # Reinitialize the EVA weights.
        # This method is used to set the LoRA weights based on the SVD components.
        if lora_rank != weight_a.shape[0]:
            raise ValueError(f"lora_rank {lora_rank} does not match weight_a shape {weight_a.shape[0]}")
        if weight_a.shape[1] != self.in_features:
            raise ValueError(f"weight_a shape {weight_a.shape} does not match in_features {self.in_features}")

        self.EVA_init(lora_rank)
        self.weight_a.data = weight_a.contiguous().to(dtype=self._get_lora_dtype())

class SVDHook:
    def __init__(
        self,
        name: str,
        n_components: int,
        sim_thresh: Union[float, torch.Tensor]
    ):
        self.name = name
        self.n_components = n_components
        self.sim_thresh = sim_thresh

        if isinstance(sim_thresh, torch.Tensor) and len(sim_thresh.shape) > 0:
            check1 = sim_thresh.size(0) == n_components or sim_thresh.size(0) == 1
            check2 = len(sim_thresh.shape) == 1
            if not check1 or not check2:
                raise ValueError("if sim_thresh is a tensor with more than 0 dimensions it must have shape (n_components,) or (1,)")

        self.svd = IncrementalPCA(n_components=n_components, copy=True, lowrank=True)

        self.indices = None
        self.converged = torch.zeros((n_components,), dtype=torch.bool)

    def __call__(self, model, input, output):
        previous_components = None
        if hasattr(self.svd, "components_"):
            previous_components = self.svd.components_.clone().detach()

        try:
            states = input.detach()
        except AttributeError:
            states = input[0].detach()
        states = states[self.indices[:, 0], self.indices[:, 1], :]

        if states.size(0) < self.n_components:
            return

        self.svd.partial_fit(states.to(torch.float32))

        if previous_components is not None:
            components = self.svd.components_
            if len(components.shape) == 1:
                components = components.reshape(1, -1)
                previous_components = previous_components.reshape(1, -1)
            # consider as converged if enough components have converged via cossim
            sim = torch.nn.functional.cosine_similarity(components, previous_components)
            self.converged = (sim >= self.sim_thresh)

def cycle(iterable):
    """Cycle through an iterable indefinitely."""
    while True:
        for item in iterable:
            yield item

def get_metric(svd, metric):
    if metric == "raw":
        return svd.explained_variance_
    elif metric == "ratio":
        return svd.explained_variance_ratio_
    elif metric == "sum":
        return svd.explained_variance_ / svd.explained_variance_.sum()
    elif metric == "max":
        return svd.explained_variance_ / svd.explained_variance_.max()

    else:
        raise ValueError(f"Invalid metric: {metric}")

def get_rank_distribution(hooks, metric, rank_budget, max_components):
    exp_vars = {k: get_metric(h.svd, metric)[:max_components] for k, h in hooks.items()}
    keys, values = zip(*[(k, c) for k, _ in hooks.items() for c in exp_vars[k]])
    idx = torch.stack(values).argsort(descending=True)
    counts = Counter([keys[i] for i in idx[:rank_budget]])
    counts = {k: counts.get(k, 0) for k in hooks.keys()}
    return counts

def compute_svd(
    model: nn.Module,
    data_loader,
    args,
    forward_func: Callable = None,
):
    if args.eva_rho < 1:
        raise ValueError("early_stop_rho must be >= 1")
    max_components = round(args.lora_rank * args.eva_rho)
    model.eval()

    hooks = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearWithEVA):
            hooks[name] = SVDHook(name, max_components, args.eva_early_stop_sim_thresh)
    rank_budget = len(hooks) * args.lora_rank

    has_converged_stats = None
    if args.eva_log_convergence_stats:
        has_converged_stats = [{
            "rank": args.lora_rank,
            "rho": args.eva_rho,
            "early_stop_sim_thresh": args.eva_early_stop_sim_thresh,
            "early_stop_redist_metric": args.eva_early_stop_redist_metric,
            "scale_by_singular_values": args.eva_scale_by_singular_values,
            "whiten": args.eva_whiten,
        }]
    
    # start svd calculation
    pbar = tqdm(enumerate(iter(cycle(data_loader))), total=args.eva_n_steps, position=0, leave=False)
    convergence_dict = {k: False for k in hooks.keys()}
    rank_dist = {k: max_components for k in hooks.keys()}

    for i, inputs in pbar:
        t0 = time.perf_counter()

        mask = inputs["labels"] != args.pad_id
        indices = torch.nonzero(mask)
        # inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}
        inputs = to_device(inputs, args.device)

        for name, hook in hooks.items():
            # Get the target module in hooks
            module = reduce(getattr, name.split("."), model)
            module._forward_hooks.clear()
            # check if all components that are needed for the rank distribution have converged
            if torch.all(hook.converged[:rank_dist[name]]):
                convergence_dict[name] = True
                continue
            convergence_dict[name] = False
            hook.indices = indices
            module.register_forward_hook(hook)

        if all(convergence_dict.values()) and i > args.eva_n_steps:
            print_rank_0("exiting - all svd components have converged.", 0)
            break
        # forward pass
        if forward_func:
            forward_func(model, inputs)
        else:
            model(**inputs)

        # in case some hooks have to skip the svd calculation because the number of tokens is less than the number of components
        if not all([hasattr(h.svd, "components_") for h in hooks.values()]):
            continue

        rank_dist = get_rank_distribution(hooks, args.eva_early_stop_redist_metric, rank_budget, max_components)

        step_time = time.perf_counter() - t0

        layer_converged = list(convergence_dict.values())
        pbar.set_description(f"{sum(layer_converged)}/{len(layer_converged)} layers have converged")

        if args.eva_log_convergence_stats:
            stats = {k: hook.converged.tolist() for k, hook in hooks.items()}
            has_converged_stats.append((stats, step_time))

    svd_dict = {}
    for name, rank in rank_dist.items():
        if rank == 0:
            continue
        hook = hooks[name]
        assert torch.all(hook.converged[:rank]) # this should never happen because we check for convergence
        u = hook.svd.components_[:rank]
        if args.eva_whiten:
            u /= hook.svd.singular_values_[:rank].sqrt().reshape(-1, 1)
        elif args.eva_scale_by_singular_values:
            s = hook.svd.singular_values_[:rank]
            s /= s.max()
            u *= s.reshape(-1, 1)
        svd_dict[name] = u

    # objects are torch tensors on the model device
    svd_dict = {k: v.cpu() for k, v in svd_dict.items()}

    return svd_dict, rank_dist, has_converged_stats

def convert_tensors_to_list(obj):
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_tensors_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tensors_to_list(elem) for elem in obj]
    else:
        return obj

def eva_reinit(
    model: nn.Module, 
    dataloader, 
    args,
    task_name: str = '',
    forward_func: Callable = None
):
    if getattr(args, "pad_id", None) is None:
        raise ValueError("pad_id must be provided for EVA reinitialization")

    svd_dict, rank_dist_dict = None, None
    with Timer() as timer:
        if args.global_rank == 0:
            print_rank_0("---> Computing SVD for EVA initialization...", args.global_rank)
            svd_dict, rank_dist_dict, has_converged_stats = compute_svd(
                model=model,
                data_loader=dataloader,
                forward_func=forward_func,
                args=args
            )

            results_to_broadcast = [svd_dict, rank_dist_dict]
            
            torch.distributed.broadcast_object_list(results_to_broadcast, src=0)

            
            save_folder = os.path.join(args.output_path, args.experiment_name)
            if task_name:
                save_folder = os.path.join(save_folder, task_name)
            ensure_directory_exists(save_folder, args.global_rank)
            with open(os.path.join(save_folder, 'rank.json'), 'w') as f:
                json.dump(convert_tensors_to_list(rank_dist_dict), f)
            with open(os.path.join(save_folder, 'svd_init.json'), 'w') as f:
                json.dump(convert_tensors_to_list(svd_dict), f)
            if has_converged_stats:
                with open(os.path.join(save_folder, 'has_converged_stats.json'), 'w') as f:
                    json.dump(convert_tensors_to_list(has_converged_stats), f)
            print_rank_0(f'---> EVA reinit artifacts saved to {save_folder}', args.global_rank)
        else:
            results_to_broadcast = [None, None]
            torch.distributed.broadcast_object_list(results_to_broadcast, src=0)
            
            svd_dict, rank_dist_dict = results_to_broadcast
            
        torch.distributed.barrier()

        print_rank_0(f"---> Re-initializing model weights...", args.global_rank)
        for name, module in model.named_modules():
            if isinstance(module, LinearWithEVA):
                if name in svd_dict and rank_dist_dict.get(name, 0) > 0:
                    module.EVA_reinit(rank_dist_dict[name], svd_dict[name])
                else:
                    module.merge_and_del()
                    if args.global_rank == 0 and name in rank_dist_dict:
                        print_rank_0(f'---> Warning: {name} was allocated, merging and deleting lora weights.', args.global_rank)

    print_rank_0(f'---> Total time consumed for EVA initialization: {timer.time_cost}.', args.global_rank)