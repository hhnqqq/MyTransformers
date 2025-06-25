import torch

sd = torch.load('', map_location='cpu')

for n, p in sd.items():
    p = p.cuda().float()
    if p.dim()==2:
        nuc_norm = torch.linalg.matrix_norm(p, ord=2)
        print(f'name:{n}, norm:{nuc_norm}')