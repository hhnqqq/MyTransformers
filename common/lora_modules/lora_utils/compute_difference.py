import torch

full_sd = torch.load('')
tdlora_sd = torch.load(r'')['model_state_dict']
lora_sd = torch.load('')['model_state_dict']


layer_names = []
direction_diffs = []
magnitude_diffs = []

# 遍历模型权重
for (tdlora_k, tdlora_v), (lora_k, lora_v) in zip(tdlora_sd.items(), lora_sd.items()):
    layer_name, weight_name = tuple(tdlora_k.rsplit('.', maxsplit=1))
    if 'weight_a' in weight_name:
        tdlora_A = tdlora_v
        lora_A = lora_v
    else:
        tdlora_B = tdlora_v
        lora_B = lora_v
        rank = tdlora_A.shape[0]
        tdlora_compute_result = torch.matmul(tdlora_B, tdlora_A)
        lora_compute_result = torch.matmul(lora_B, lora_A)
        full_weight = full_sd[layer_name + '.weight']
        
        # 计算方向更新差异
        tdlora_cosine_similarity = torch.nn.functional.cosine_similarity(tdlora_compute_result + full_weight, full_weight, dim=0)
        tdlora_direction_update = torch.mean(torch.ones(full_weight.shape[1]).to(tdlora_cosine_similarity.device) - tdlora_cosine_similarity)
        
        lora_cosine_similarity = torch.nn.functional.cosine_similarity(lora_compute_result + full_weight, full_weight, dim=0)
        lora_direction_update = torch.mean(torch.ones(full_weight.shape[1]).to(lora_cosine_similarity.device) - lora_cosine_similarity)
        
        direction_diff = (tdlora_direction_update - lora_direction_update).item()
        
        # 计算大小更新差异
        tdlora_magnitude_update = torch.linalg.norm(tdlora_compute_result + full_weight, dim=1) - torch.linalg.norm(full_weight, dim=1)
        lora_magnitude_update = torch.linalg.norm(lora_compute_result + full_weight, dim=1) - torch.linalg.norm(full_weight, dim=1)
        
        magnitude_diff = (torch.mean(tdlora_magnitude_update) - torch.mean(lora_magnitude_update)).item()
        
        # 存储结果
        layer_names.append(layer_name)
        direction_diffs.append(direction_diff)
        magnitude_diffs.append(magnitude_diff)

print(layer_names, direction_diffs, magnitude_diffs)