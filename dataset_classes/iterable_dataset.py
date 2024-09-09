import json
import math
import random
from typing import Optional

from dataset_classes import BaseDataset
from torch.utils.data import IterableDataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry

@registry.register_dataset('iterable')
class BaseIterableDataset(IterableDataset, BaseDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Optional[int] = None,
        global_rank: int = 0,
        meta_prompt: str = '',
        prefix: str = 'Q:',
        postfix: str = 'A:',
        shuffle: bool = False,
        num_dp_ranks: Optional[int] = None,
        dp_rank: Optional[int] = None,
        cal_metric_pos: Optional[int] = None,
        encode_single_gene: bool = False,
        *args,
        **kwargs
    ):
        BaseDataset.build_data(
        self,
        data_path,
        tokenizer,
        max_len,
        max_src_len,
        mode,
        read_nums,
        global_rank,
        meta_prompt,
        prefix,
        postfix,
        cal_metric_pos,
        encode_single_gene
    )
        self.shuffle = shuffle
        if num_dp_ranks and dp_rank is not None:
            self.dp = True
            read_nums_per_rank = math.ceil(self.read_nums / num_dp_ranks)
            self.start = read_nums_per_rank * dp_rank
            self.end = min(read_nums_per_rank * (dp_rank + 1), self.read_nums)
            print_rank_0(f'--->global rank:{self.global_rank} read range [{self.start}:{self.end}]', self.global_rank, force_print=True)
        else:
            self.start = 0
            self.end = self.read_nums
        if shuffle:
            # Make sure random seed has been set
            dataset_rng = random.Random(42)
            read_indices = list(range(self.read_nums))
            if read_nums is not None:
                read_indices = dataset_rng.sample(read_indices, read_nums)
            else:
                dataset_rng.shuffle(read_indices)
            self.read_indices = read_indices[self.start:self.end]
        
    def _load_sample(self, i, line):
        try:
            sample = json.loads(line.strip())
        except:
            sample = line.strip()
            if i==0:
                print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
        return sample
    
    def __iter__(self):
        if self.shuffle:
            print_rank_0('--->Dataset shuffle is enabled', self.global_rank)
            yield from self._shuffle_iter()
        else:
            with open(self.data_path, "r", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if self.start <= i < self.end:
                        sample = self._load_sample(i, line)
                        yield BaseDataset.process_sample(self, sample)
                
    def _shuffle_iter(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        for read_idx in self.read_indices:
            line = lines[read_idx]
            if line:
                sample = self._load_sample(read_idx, line)
                yield BaseDataset.process_sample(self, sample)
                
    def __len__(self):       
        return self.read_nums
    

# if __name__ == "__main__":
#     # Test example.
#     import os
#     import torch
#     from common.utils import DataCollator, set_random_seed
#     from torch.utils.data import DataLoader
#     from model.tokenizer import Llama3Tokenizer
#     from deepspeed.utils import RepeatingLoader

#     set_random_seed(114514)
#     os.environ['NO_LOG_FILE'] = 'true'
#     file_path = '/home/bingxing2/ailab/group/ai4bio/public/qatext/dna-train.jsonl'
#     tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model'
#     tokenizer = Llama3Tokenizer(tokenizer_path)
#     data_collator = DataCollator(tokenizer)

#     iterable_dataset = IterableDataset(file_path,
#                                        tokenizer,
#                                        max_len=650,
#                                        max_src_len=600,
#                                        mode='sft',
#                                        prefix='<|start_header_id|>user<|end_header_id|>\n\n',
#                                        postfix='<|start_header_id|>assistant<|end_header_id|>\n\n',
#                                        meta_prompt='<|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable and helpful biology assistant. Please answer my biology sequence-related questions in a clear and concise manner.',
#                                        shuffle=True)
        
#     g = torch.Generator()
#     dataloader = RepeatingLoader(DataLoader(iterable_dataset,
#                             collate_fn=data_collator,
#                             shuffle=False,
#                             drop_last=True,
#                             batch_size=8,
#                             generator=g))
    
#     for i, data in enumerate(dataloader):
#         pass