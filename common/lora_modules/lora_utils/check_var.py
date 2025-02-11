import math
import torch
path = '/ailab/user/hehaonan/checkpoint/8gpu_math_base/math_tdlora_random_fp32/final.ckpt'
lora_rank = 8
lora_alpha = 16

def subspace_similarity(U1, U2):
    _, s, _ = torch.svd_lowrank(U1.T @ U2, lora_rank)
    return s 

a = torch.randn((4096, 4096))
m, n = a.shape
stable_scaling =(8e-2 / math.sqrt(lora_rank/n)) * (math.sqrt(lora_rank))
# stable_scaling = 16
torch.nn.init.kaiming_uniform_(a, 5**0.5)
# print(torch.var(a))
print(torch.linalg.matrix_norm(a))

weight_a = torch.randn((lora_rank, n))
torch.nn.init.kaiming_uniform_(weight_a, 5**0.5)
weight_b = torch.randn((m, lora_rank))
torch.nn.init.kaiming_uniform_(weight_b, 5**0.5)
print(torch.linalg.matrix_norm(torch.matmul(weight_b, weight_a)* lora_alpha /lora_rank))
AT = weight_a.T
AAT = torch.matmul(weight_a, AT)
print(torch.linalg.matrix_norm(AT))


AAT_inv = torch.linalg.pinv(AAT + 1e-8 * torch.eye(lora_rank))
AAT_inv_AT = torch.matmul(AT, AAT_inv)
print(torch.linalg.matrix_norm(AAT_inv_AT))

weight_b = torch.matmul(a, AAT_inv_AT) * (stable_scaling / lora_alpha)
print(torch.linalg.matrix_norm(weight_b))
print(torch.var(weight_a), torch.var(weight_b))
# rank越大，乘出来的结果就越大，就会导致不稳定
G_new = torch.matmul(weight_b, weight_a)*(lora_alpha/math.sqrt(lora_rank))
print(torch.linalg.matrix_norm(G_new))
# print(subspace_similarity(a,G_new))


# 适合和lora+一起用
# sd = torch.load(path, map_location='cpu')
# sd = sd['model_state_dict']

# a_sd = {n:p for n,p in sd.items() if 'weight_a' in n}
# b_sd = {n:p for n,p in sd.items() if 'weight_b' in n}

# a_var =  torch.tensor([torch.var(p) for p in a_sd.values()])
# b_var =  torch.tensor([torch.var(p) for p in b_sd.values()])
# print(torch.mean(a_var))
# print(torch.mean(b_var))



# print(torch.var(a))
# U,S,V = torch.svd_lowrank(a, 32, 4)
# V = V.T
# stable_gamma = 16
# B = U[:, lora_rank:2 * lora_rank]
# A = V[:lora_rank, :]
# print(torch.var(A), torch.var(B))
# A = A * m**0.25 / stable_gamma**0.5
# B = B * m**0.25 / stable_gamma**0.5

# print(torch.var(A), torch.var(B))
