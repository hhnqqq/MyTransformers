import torch
import pickle
import torch.distributed as dist
from common.utils.utils import reduce_tensor

def get_record_gradient_hook(model, world_size=1, rank=0):
    def record_gradient_hook(grad):
        torch.cuda.synchronize()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                if world_size > 1:
                    p.grad = reduce_tensor(p.grad, world_size)
                if rank == 0:
                    if not hasattr(p, 'grad_stored'):
                        p.grad_stored = p.grad.detach().cpu()
                        p.iters = 1
                    else:
                        p.grad_stored += p.grad.detach().cpu()
                        p.iters += 1
                p.grad = None  # Clear GPU gradient immediately
        return grad
    return record_gradient_hook


def broadcast_object(args, broadcast_data=None):
    dist.barrier()
    
    if args.global_rank == 0:
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
    
    return broadcast_data