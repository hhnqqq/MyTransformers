import json
import math
import random
import bisect
from typing import Optional, List

from collections.abc import Iterable
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry
from dataset_classes.base_dataset import BaseDataset
from dataset_classes.iterable_dataset import BaseIterableDataset

@registry.register_dataset('concat')
class ConcatDataset(BaseDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:  
        datasets (sequence): List of datasets to be concatenated.
    """

    @staticmethod
    def cumsum(sequence, weights):
        r, s = [], 0
        for i, e in enumerate(sequence):
            l = int(len(e) * weights[i])
            r.append(l + s)
            s += l
        return r
    
    @classmethod
    def validate_data_paths(cls, data_paths):
        if not isinstance(data_paths, Iterable):
            raise ValueError('data_paths required to be Iterable in ConcatDataset.')
        if len(data_paths) == 0:
            raise ValueError('datasets should not be an empty iterable in ConcatDataset.')

    def __init__(
        self,
        data_paths: List[str],
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Optional[int] = None,
        global_rank: int=0,
        meta_prompt: str ='',
        prefix: str='Q:',
        postfix: str='A:',
        cal_metric_pos: Optional[int] = None,
        encode_single_gene: bool = False,
        weights: Optional[List[int]] = None,
        *args,
        **kwargs
    ):
        self.build_data(
        data_paths, # Avoid error in the base dataset class.
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
        encode_single_gene,
        weights
    )
        self.process_data_file()

    def build_data(self, 
        data_paths: str, 
        tokenizer: BaseTokenizer, 
        max_len: int, 
        max_src_len: int, 
        mode: str = 'pretrain', 
        read_nums: int | None = None, 
        global_rank: int = 0, 
        meta_prompt: str = '', 
        prefix: str = 'Q:', 
        postfix: str = 'A:', 
        cal_metric_pos: int | None = None, 
        encode_single_gene: bool = False,
        weights: Optional[List[int]] = None):
        temp_read_nums = read_nums
        self.validate_data_paths(data_paths)
        super().build_data(data_paths[0], # Avoid error in the base dataset class.
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
                           encode_single_gene)
        self.datasets = [open(data_path, 'r').readlines() for data_path in data_paths]
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sum = self.cumsum(self.datasets, self.weights)
        if temp_read_nums is None:
            self.read_nums = self.cumulative_sum[-1]


    def process_data_file(self):
        count = 0
        stop = False  # 添加一个标志变量

        for weight, dataset_lines in zip(self.weights, self.datasets):
            for repeat in range(weight):
                for i, line in enumerate(dataset_lines):
                    try:
                        sample = json.loads(line.strip())
                    except:
                        sample = line.strip()
                        if i == 0:
                            print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
                    
                    self.all_data.append(self.process_sample(sample))
                    count += 1
                    if count >= self.read_nums:
                        stop = True
                        break
                if stop: 
                    break 
            if stop: 
                break
            print_rank_0(f'--->train_tokens:{self._get_post_fix(self.train_token_count)}', self.global_rank)

    def __len__(self):
        return self.read_nums
    
@registry.register_dataset('concat_iterable')
class IterableConcatDataset(BaseIterableDataset, ConcatDataset):
    
    def __init__(
        self,
        data_paths: str,
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
        weights: Optional[List[int]] = None,
        *args,
        **kwargs):
        ConcatDataset.build_data(
        self,
        data_paths,
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
        encode_single_gene,
        weights
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
    
    def __iter__(self):
        for read_idx in self.read_indices:
            dataset_index = bisect.bisect_right(self.cumulative_sum, read_idx)
            dataset = self.datasets[dataset_index]

            line = dataset[read_idx-self.cumulative_sum[dataset_index]]
            if line:
                sample = BaseIterableDataset._load_sample(self, read_idx, line)
                yield BaseIterableDataset.process_sample(self, sample)


if __name__ == "__main__":
    # Test example.
    import os
    import torch
    from common.utils import DataCollator, set_random_seed
    from torch.utils.data import DataLoader
    from model.tokenizer import Llama3Tokenizer
    from deepspeed.utils import RepeatingLoader

    set_random_seed(114514)
    os.environ['NO_LOG_FILE'] = 'true'
    file_path = ['/home/bingxing2/ailab/group/ai4bio/public/qatext/dna-train.jsonl',
                 '/home/bingxing2/ailab/group/ai4bio/public/qatext/dna-test.jsonl']
    tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model'
    tokenizer = Llama3Tokenizer(tokenizer_path)
    data_collator = DataCollator(tokenizer)

    iterable_dataset = IterableConcatDataset(file_path,
                                       tokenizer,
                                       max_len=650,
                                       max_src_len=600,
                                       mode='sft',
                                       prefix='<|start_header_id|>user<|end_header_id|>\n\n',
                                       postfix='<|start_header_id|>assistant<|end_header_id|>\n\n',
                                       meta_prompt='<|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable and helpful biology assistant. Please answer my biology sequence-related questions in a clear and concise manner.',
                                       shuffle=True)
        
    g = torch.Generator()
    dataloader = RepeatingLoader(DataLoader(iterable_dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            batch_size=8,
                            generator=g))
    
    for i, data in enumerate(dataloader):
        print(data)