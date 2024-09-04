import json
import math
import random
from typing import Optional

from dataset_classes import BaseDataset
from torch.utils.data import IterableDataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0, is_seed_set
from common.registry import registry

@registry.register_dataset('iterable')
class IterableDataset(IterableDataset, BaseDataset):
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
        read_indices = list(range(self.read_nums))
        self.read_indices = read_indices[self.start:self.end]
        # Make sure random seed has been set
        assert is_seed_set()
        random.shuffle(self.read_indices)
        
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