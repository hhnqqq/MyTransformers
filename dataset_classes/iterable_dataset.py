import json
import math
import numpy as np
from typing import Optional

from dataset_classes import BaseDataset
from torch.utils.data import IterableDataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry

@registry.register_dataset('iterable')
class BaseIterableDataset(IterableDataset, BaseDataset):
    """
    BaseIterableDataset is an iterable dataset class that inherits from both IterableDataset and BaseDataset.
    It is registered under the name 'iterable' in the registry.


    Args:
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        num_dp_ranks (Optional[int], optional): Number of data parallel ranks. Defaults to None.
        dp_rank (Optional[int], optional): Data parallel rank. Defaults to None.
        start_step (int): Start step for training. It is useful when training on previous checpoint. This value should equals to `micro_batch_size*before_micro_step`
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Note:
        For detailed explanations of other parameters, please refer to the parent class BaseDataset.
    """
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
        padding: bool = True,
        start_step: int = 0,
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
        encode_single_gene,
        padding
    )
        self.init_parallel_and_shuffle(shuffle, num_dp_ranks, dp_rank, read_nums, start_step)

    def init_parallel_and_shuffle(self, shuffle, num_dp_ranks, dp_rank, read_nums, start_step):
        """
        Initialize parallel processing settings for the dataset.


        Args:
            shuffle (bool): Whether to shuffle the dataset.
            num_dp_ranks (Optional[int]): Number of data parallel ranks.
            dp_rank (Optional[int]): Data parallel rank.
            read_nums (Optional[int]): Number of samples to read.
        """
        self.shuffle = shuffle
        self.start_step=start_step
        if num_dp_ranks and dp_rank is not None:
            """
            E.g.
                Assume data parallel group [0, 1, 2, 3], training with a dataset composed of 1000 samples.
                In this case, read_nums_per_rank=1000/4=250 
                rank0 read range is [0:250], rank1 read range is [250:500] and so on
            """
            # Calculate the number of samples to read per rank
            read_nums_per_rank = math.ceil(self.read_nums / num_dp_ranks)
            self.start = read_nums_per_rank * dp_rank
            self.end = min(read_nums_per_rank * (dp_rank + 1), self.read_nums)
            print_rank_0(f'--->global rank:{self.global_rank} read range [{self.start}:{self.end}]', self.global_rank, force_print=True)
        else:
            # If no data parallel ranks are specified, read all samples
            self.start = 0
            self.end = self.read_nums
        self.init_shuffle(shuffle, read_nums)

    def init_shuffle(self, shuffle, read_nums):
        """
        Initialize read indices for shuffle reading.

        E.g.
            Consider a dataset cotains 10 lines trained by 2 data parallel ranks.
            Each rank read 5 samples, the initial read indices is [0,1,2,3,4,5,6,7,8,9]
            After shuffle, the read indices of rank will be [5,6,0,7,3,2,4,9,1,8]
            Therefore, the read order of rank 2 will be [2,4,9,1,8]
        """
        if shuffle:
            # Make sure random seed has been set
            print_rank_0(f'--->Dataset shuffle is enabled', self.global_rank)
            dataset_rng = np.random.default_rng(42)
            if read_nums is not None:
                # Randomly select indices for reading
                read_indices = dataset_rng.choice(self.line_count, size=read_nums, replace=False)
            else:
                """
                Equals to:
                    read_indices = list(range(self.read_nums))
                    random.shuffle(read_indices)
                This code will shuffle the read indices first, then split them to respective parallel rank.
                """
                read_indices = dataset_rng.permutation(self.read_nums)
        else:
            # If no shuffling, use sequential indices
            read_indices = list(range(self.read_nums))

        # Slice the read indices based on the calculated start and end positions
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
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # E.g. latest checkpoint is step_10000.ckpt.
        # And micro batch size used before is 4
        # Start step should be 4*10000 = 40000.
        # This time the 40001th data should be read.
        step = self._get_start_step()
        # Equals to for read_idx in self.read_indices:
        while step <= len(self.read_indices):
            read_idx = self.read_indices[step]
            line = lines[read_idx]
            step += 1
            if line:
                sample = self._load_sample(read_idx, line)
                yield BaseDataset.process_sample(self, sample)
                
    def _get_start_step(self):
        return self.start_step % len(self.read_indices) + 1
    
    def __len__(self):       
        return self.read_nums
    

    # def __iter__(self):
    #     if self.shuffle:
    #         print_rank_0('--->Dataset shuffle is enabled', self.global_rank)
    #         yield from self._shuffle_iter()
    #     else:
    #         with open(self.data_path, "r", encoding="utf-8") as fh:
    #             for i, line in enumerate(fh):
    #                 if self.start <= i < self.end:
    #                     sample = self._load_sample(i, line)
    #                     yield BaseDataset.process_sample(self, sample)
             

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
    file_path = '/home/bingxing2/ailab/group/ai4bio/public/qatext/dna-train.jsonl'
    tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama3_tokenizer.model'
    tokenizer = Llama3Tokenizer(tokenizer_path)
    data_collator = DataCollator(tokenizer)

    iterable_dataset = IterableDataset(file_path,
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
        pass