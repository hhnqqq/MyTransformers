import json
import math
import bisect
import itertools
import numpy as np
from typing import Optional, Union, List

from collections.abc import Iterable
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry
from common.utils import print_rank_0
from dataset_classes.base_dataset import BaseDataset, DatasetConfig
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
        dataset_config: DatasetConfig,
        read_nums: Optional[int] = None,
        global_rank: int=0,
        weights: Optional[List[int]] = None,
        *args,
        **kwargs
    ):
        self.build_data(
        data_paths, # Avoid error in the base dataset class.
        tokenizer,
        dataset_config,
        read_nums,
        global_rank,
        weights
    )
        self.process_data_file()

    def _process_data_format_encode(self, input_data):
        if isinstance(input_data, list) and len(input_data) == len(self.datasets):
            return {idx: BaseDataset._process_data_format_encode(self, item) if item else [] for idx, item in enumerate(input_data)}
        else:
            input_data = input_data[0] if isinstance(input_data, list) else input_data
            return super()._process_data_format_encode(input_data)


    def build_data(self, 
        data_paths: List[str], 
        tokenizer: BaseTokenizer, 
        dataset_config: DatasetConfig,
        read_nums: int | None = None, 
        global_rank: int = 0,
        weights: Optional[List[int]] = None):
        if isinstance(data_paths, str):
            data_paths = [data_paths]
        self.validate_data_paths(data_paths)
        self.datasets = [MmapDataset(data_path) for data_path in data_paths]
        super().build_data(data_paths, # Avoid error in the base dataset class.
                           tokenizer, 
                           dataset_config,
                           read_nums, 
                           global_rank)
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

        for dataset_index, weight, dataset_lines in enumerate(zip(self.weights, self.datasets)):
            for repeat in range(weight):
                for i, line in enumerate(dataset_lines):
                    try:
                        sample = json.loads(line.strip())
                    except:
                        sample = line.strip()
                        if i == 0:
                            print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
                    
                    sample = {"sample":sample, "dataset_index":dataset_index}
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

    def preprocess_sample(self, sample):
        if isinstance(sample, dict) and "input_ids" in sample.keys():
            self.preprocess_sample_from_ids(sample)
        else:
            dataset_index, input_text, output_text = self._extract_texts(sample)
            input_ids = self._process_text(input_text, dataset_index)
            return input_text, output_text, input_ids, []

    def _extract_texts(self, sample):
        dataset_index = sample['dataset_index']
        sample = sample['sample']
        input_text, output_text = super()._extract_texts(sample)
        return dataset_index, input_text, output_text
    
    def _process_text(self, input_text, dataset_index):
        encoded_ids = self._encode_text(input_text)
        # Make sure that the bos id always be the first id of input ids.
        meta_prompt = self.meta_prompt[dataset_index] if isinstance(self.meta_prompt, dict) else self.meta_prompt
        prefix = self.prefix[dataset_index] if isinstance(self.prefix, dict) else self.prefix
        postfix = self.postfix[dataset_index] if isinstance(self.postfix, dict) else self.postfix
        input_ids = [self.tokenizer.bos_id] + meta_prompt + prefix + encoded_ids + postfix
        return input_ids
    
    def __len__(self):
        return self.read_nums
    

    
@registry.register_dataset('concat_iterable')
class IterableConcatDataset(BaseIterableDataset, ConcatDataset):
    
    def __init__(
        self,
        data_paths: List[str],
        tokenizer: BaseTokenizer,
        dataset_config: DatasetConfig,
        read_nums: Optional[int] = None,
        global_rank: int = 0,
        shuffle: bool = False,
        num_dp_ranks: Optional[int] = None,
        dp_rank: Optional[int] = None,
        weights: Optional[List[int]] = None,
        read_sequential: bool = False,
        seed: int = 42, 
        start_step: int = 0,
        *args,
        **kwargs):
        ConcatDataset.build_data(
        self,
        data_paths,
        tokenizer,
        dataset_config,
        read_nums,
        global_rank,
        weights
    )
        # ConcatDataset originally read sequential, so this feature only needed in IterableConcatDataset.
        self.read_sequential = read_sequential
        self.init_parallel_and_shuffle(shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step)

    def init_parallel_and_shuffle(self, shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step):
        if self.read_sequential:
            self.init_sequential_indices(shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step)
        else:
            super().init_parallel_and_shuffle(shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step)

    def init_sequential_indices(self, shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step):
        assert read_nums is None, "sequential read dataset is not competible with given a `read_nums`"
        self.shuffle = shuffle
        self.start_step = start_step
        read_indices_per_datasets = [
            list(range(0 if i == 0 else self.cumulative_sum[i-1], count))
            for i, count in enumerate(self.cumulative_sum)
        ]

        if num_dp_ranks and dp_rank is not None:
            for i, indices in enumerate(read_indices_per_datasets):
                nums_per_rank = math.ceil(len(indices) / num_dp_ranks)
                start = dp_rank * nums_per_rank
                end = min((dp_rank + 1) * nums_per_rank, len(indices))
                read_indices_per_datasets[i] = indices[start:end]

        self.init_sequential_shuffle(read_indices_per_datasets, seed)

    def init_sequential_shuffle(self, read_indices_per_datasets, seed):
        if self.shuffle:
            dataset_rng = np.random.default_rng(seed)
            read_indices_per_datasets = [dataset_rng.permutation(indices) for indices in read_indices_per_datasets]
        self.read_indices = list(itertools.chain(*read_indices_per_datasets))
        print_rank_0(f"--->Using seed {seed} for IterableConcatDataset, top 5 read indices of each dataset are: {[indices[:5] for indices in read_indices_per_datasets]}", self.global_rank)

    def __iter__(self):
        # Example for start step see BaseIterableDataset.
        step = self._get_start_step()

        while step < len(self.read_indices):
            read_idx = self.read_indices[step]
            # Determine the dataset to be chosen based on the read index.
            # For example, with three datasets of sizes: [1000, 2000, 3000], the cumulative sum is [1000, 3000, 6000].
            # If the read index is 1500, the dataset_index will be 1, since 1500 falls between 1000 and 3000.
            # pre_length is set to self.cumulative_sum[1-1] = 1000.
            # Thus, the 500th data point in the 2sd dataset will be accessed.
            dataset_index = bisect.bisect_right(self.cumulative_sum, read_idx)
            dataset = self.datasets[dataset_index]
            pre_length = 0 if dataset_index == 0 else self.cumulative_sum[dataset_index-1]
            adjusted_read_idx = read_idx - pre_length
            if self.weights[dataset_index] > 1:
                adjusted_read_idx = adjusted_read_idx % len(dataset)
            line = dataset[adjusted_read_idx]
            step+=1
            if line:
                sample = BaseIterableDataset._load_sample(self, read_idx, line)
                sample = {"sample":sample, "dataset_index":dataset_index}
                yield ConcatDataset.process_sample(self, sample)


    def __len__(self):       
        return self.read_nums


if __name__ == "__main__":
    # Test example.
    import os
    import torch
    from common.utils import DataCollator, set_random_seed
    from torch.utils.data import DataLoader
    from model.tokenizer import Llama3Tokenizer
    from dataset_classes import RepeatingLoader
    from dataset_classes.packing_dataset import IterablePackingDataset

    set_random_seed(114514)
    os.environ['NO_LOG_FILE'] = 'true'
    file_path = [
        ]
    # meta_prompt = ["<|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable and helpful biology assistant. Please answer my biology sequence-related questions in a clear and concise manner. For regression task, please return a number. <|eot_id|>",
    # "<|start_header_id|>system<|end_header_id|>\n\nYou are a highly knowledgeable AI assistant specializing in biology, particularly in sequence-related topics. Your primary task is to provide clear, accurate, and comprehensive answers to biology questions. When analyzing and interpreting sequences, ensure to provide step-by-step explanations to make your responses natural and easy to understand. Engage with the user by asking clarifying questions if needed and offer detailed insights into the biological sequences. <|eot_id|>"]
    meta_prompt = "<|start_header_id|>system<|end_header_id|>\n\nYou are a knowledgeable and helpful biology assistant. Please answer my biology sequence-related questions in a clear and concise manner. For regression task, please return a number. <|eot_id|>"
    tokenizer_path = ''
    tokenizer = Llama3Tokenizer(tokenizer_path)
    data_collator = DataCollator(tokenizer)

    iterable_dataset = IterableConcatDataset(file_path,
                                       tokenizer,
                                       max_len=1024,
                                       max_src_len=1024,
                                       mode='pretrain',
                                       prefix=None,
                                       postfix=None,
                                       meta_prompt=meta_prompt,
                                       shuffle=True,
                                       padding=False,
                                       weights=None,
                                       read_sequential=False)
        
    iterable_dataset = IterablePackingDataset(iterable_dataset, 1100)
    g = torch.Generator()
    dataloader = RepeatingLoader(DataLoader(iterable_dataset,
                            collate_fn=data_collator,
                            shuffle=False,
                            drop_last=True,
                            batch_size=5,
                            generator=g))
    
    for i, data in enumerate(dataloader):
        """
        Every iteration, the read indices keep same.
        [2024-09-21 15:01:16,155] [INFO] --->Start a new iteration for repeating loader for rank 0.
        [ 396166 3567548  411401 3017246 2023009 1995983 3214520  434110  928674 3957707]
        [2024-09-21 15:01:16,169] [INFO] --->Start a new iteration for repeating loader for rank 0.
        [ 396166 3567548  411401 3017246 2023009 1995983 3214520  434110  928674 3957707]"""
        pass
        # print(dataloader.train_token_count)
        # print(data)