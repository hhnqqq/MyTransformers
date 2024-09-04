import json
from typing import Optional, List

from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry
from dataset_classes import BaseDataset

registry.register_dataset('concat')
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
        encode_single_gene
    )
        self.process_data_file()
        super(ConcatDataset, self).__init__()
        assert len(data_paths) > 0, 'datasets should not be an empty iterable'
        self.datasets = [open(data_path, 'r').readlines() for data_path in data_paths]
        if weights is None:
            self.weights = [1] * len(self.datasets)
        else:
            self.weights = weights
        self.cumulative_sizes = self.cumsum(self.datasets, self.weights)

    def process_data_file(self):
        for weight, dataset_lines in zip(self.weights, self.datasets):
            for repeat in range(weight):
                read_num_total = 0
                for i, line in enumerate(dataset_lines):
                    read_num_total += 1
                    if read_num_total < self.read_nums:
                        try:
                            sample = json.loads(line.strip())
                        except:
                            sample = line.strip()
                            if i==0:
                                print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
                        self.all_data.append(self.process_sample(sample))
                    else:
                        break
        print_rank_0(f'--->train_tokens:{self._get_post_fix(self.train_token_count)}', self.global_rank)


    def __len__(self):
        return self.cumulative_sizes[-1]
