import json
import math
import bisect
import itertools
import numpy as np
from typing import Optional, List

from collections.abc import Iterable
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry
from common.utils import print_rank_0
from dataset_classes.base_dataset import BaseDataset
from dataset_classes.iterable_dataset import BaseIterableDataset
from dataset_classes.dataset_tools import get_line_count, MmapDataset


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
    def cumsum(dataset_paths, weights):
        r, s = [], 0
        for i, e in enumerate(dataset_paths):
            dataset_len = get_line_count(e)
            l = int(dataset_len * weights[i])
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
        padding: bool = True,
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
        padding,
        weights
    )
        self.process_data_file()

    def build_data(self, 
        data_paths: List[str], 
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
        padding: bool = True,
        weights: Optional[List[int]] = None):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.validate_data_paths(data_paths)
        super().build_data(data_paths, # Avoid error in the base dataset class.
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
                           padding)
        self.datasets = [MmapDataset(data_path) for data_path in data_paths]
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            print_rank_0(f'Using dataset weights: {weights}', self.global_rank)
            self.weights = weights
        self.cumulative_sum = self.cumsum(data_paths, self.weights)
        self.line_count = self.cumulative_sum[-1]
        if read_nums is None:
            # reset read nums to the last item of cumulative sum
            # None -> cumulative sum. else self.read_nums = read_nums
            self.read_nums = self.line_count
        else:
            self.read_nums = read_nums


    def process_data_file(self):
        count = 0
        stop = False  

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
        data_paths: List[str],
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
        padding: bool = True,
        weights: Optional[List[int]] = None,
        read_sequential: bool = False,
        start_step: int = 0,
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
        padding,
        weights
    )
        # ConcatDataset originally read sequential, so this feature only needed in IterableConcatDataset.
        self.read_sequential = read_sequential
        self.init_parallel_and_shuffle(shuffle, num_dp_ranks, dp_rank, read_nums, start_step)

    def init_parallel_and_shuffle(self, shuffle, num_dp_ranks, dp_rank, read_nums, start_step):
        if self.read_sequential:
            self.init_sequential_indices(shuffle, num_dp_ranks, dp_rank, read_nums, start_step)
        else:
            super().init_parallel_and_shuffle(shuffle, num_dp_ranks, dp_rank, read_nums, start_step)

    def init_sequential_indices(self, shuffle, num_dp_ranks, dp_rank, read_nums, start_step):
        assert read_nums is None, "sequential read dataset is not competible with given a `read_nums`"
        self.shuffle = shuffle
        self.start_step = start_step
        read_indices_per_datasets = [
            list(range(0 if i == 0 else self.cumulative_sum[i-1], count))
            for i, count in enumerate(self.cumulative_sum)
        ]

        if self.num_dp_ranks and self.dp_rank is not None:
            for i, indices in enumerate(read_indices_per_datasets):
                nums_per_rank = math.ceil(len(indices) / num_dp_ranks)
                start = dp_rank * nums_per_rank
                end = min((self.dp_rank + 1) * nums_per_rank, len(indices))
                read_indices_per_datasets[i] = indices[start:end]

        self.init_sequential_shuffle(read_indices_per_datasets)

    def init_sequential_shuffle(self, read_indices_per_datasets):
        if self.shuffle:
            dataset_rng = np.random.default_rng(42)
            read_indices_per_datasets = [dataset_rng.permutation(indices) for indices in read_indices_per_datasets]
        self.read_indices = list(itertools.chain(*read_indices_per_datasets))

    def __iter__(self):
        # Example for start step see BaseIterableDataset.
        step = self._get_start_step()

        while step <= len(self.read_indices):
            read_idx = self.read_indices[step]
            # Determine the dataset to be chosen based on the read index.
            # For example, with three datasets of sizes: [1000, 2000, 3000], the cumulative sum is [1000, 3000, 6000].
            # If the read index is 1500, the dataset_index will be 1, since 1500 falls between 1000 and 3000.
            # pre_length is set to self.cumulative_sum[1-1] = 1000.
            # Thus, the 500th data point in the 2sd dataset will be accessed.
            dataset_index = bisect.bisect_right(self.cumulative_sum, read_idx)
            dataset = self.datasets[dataset_index]

            pre_length = pre_length = 0 if dataset_index == 0 else self.cumulative_sum[dataset_index-1]
            adjusted_read_idx = read_idx - pre_length
            if self.weights[dataset_index] > 1:
                adjusted_read_idx = adjusted_read_idx % len(dataset)
            line = dataset[adjusted_read_idx]
            step+=1
            if line:
                sample = BaseIterableDataset._load_sample(self, read_idx, line)
                yield BaseIterableDataset.process_sample(self, sample)


    def __len__(self):       
        return self.read_nums


if __name__ == "__main__":
    # Test example.
    import os
    import torch
    from common.utils import DataCollator, set_random_seed
    from torch.utils.data import DataLoader
    from model.tokenizer import Llama3Tokenizer
    from deepspeed.utils import RepeatingLoader
    from dataset_classes.packing_dataset import IterablePackingDataset

    set_random_seed(114514)
    os.environ['NO_LOG_FILE'] = 'true'
    file_path = ["/home/bingxing2/ailab/group/ai4bio/public/multi-omics/RNA/pretraining/rnalm/human/nocoding/noncoding_rna_human_single_tag.txt",
                "/home/bingxing2/ailab/group/ai4bio/public/multi-omics/protein/pretrain/uniref50_2m.txt",
                "/home/bingxing2/ailab/group/ai4bio/renyuchen/pretraining_data/human8k/GRCh38_2k.txt"]
    tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model'
    tokenizer = Llama3Tokenizer(tokenizer_path)
    data_collator = DataCollator(tokenizer)

    iterable_dataset = IterableConcatDataset(file_path,
                                       tokenizer,
                                       max_len=1100,
                                       max_src_len=1100,
                                       mode='pretrain',
                                       prefix=None,
                                       postfix=None,
                                       meta_prompt=None,
                                       shuffle=True,
                                       padding=False,
                                       weights=[2,1,1])
        
    iterable_dataset = IterablePackingDataset(iterable_dataset, 1100)
    g = torch.Generator()
    dataloader = RepeatingLoader(DataLoader(iterable_dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            batch_size=8,
                            generator=g))
    
    for i, data in enumerate(dataloader):
        # pass
        print(data)