# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""
Origin PackingDataset code from: https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/data/concatenator.py
Usage example: https://github.com/meta-llama/llama-recipes/tree/main/recipes/quickstart/finetuning/datasets

When using packed datasets, different samples are concatenated into a fixed-size chunk.
This strategy helps to avoid excessive padding IDs in training data, especially when sample lengths vary widely.
"""

import torch
from tqdm import tqdm

from torch.utils.data import Dataset, IterableDataset


class PackingDataset(Dataset):
    def __init__(self, dataset: Dataset, chunk_size: int = 4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []

        buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            }

        for sample in tqdm(self.dataset, desc="Packing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)
    
class IterablePackingDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, chunk_size: int = 4096):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.buffer = {
            "input_ids": torch.LongTensor([]),
            # "attention_mask": [],
            "labels": torch.LongTensor([]),
            # "cal_metric_pos": []
            }
        
    def __iter__(self):
        for sample in self.dataset:
            for k, v in self.buffer.items():
                sample_val = sample.get(k)
                if sample_val is not None:
                    if isinstance(v, torch.Tensor):
                        if not isinstance(sample_val, torch.Tensor):
                            sample_val = torch.tensor(sample_val)
                        self.buffer[k] = torch.cat((v, sample_val), dim=-1)
                    else:
                        self.buffer[k] += sample_val
                # If sample_val is None, skip updating the buffer for this key

            buffer_values = [v for v in self.buffer.values() if v is not None]
            buffer_length = len(next(iter(buffer_values))) if buffer_values else 0
            while buffer_length > self.chunk_size:
                chunk = {k: v[:self.chunk_size] for k, v in self.buffer.items() if v is not None}
                self.buffer = {k: v[self.chunk_size:] for k, v in self.buffer.items() if v is not None}
                yield chunk

                buffer_values = [v for v in self.buffer.values() if v is not None]
                buffer_length = len(next(iter(buffer_values))) if buffer_values else 0


    def __len__(self):
        return len(self.dataset)