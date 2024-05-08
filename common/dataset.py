# @Author: haonan he
# @modified by: zhijian jiang
import json
import torch
import random
from tqdm import tqdm
from typing import Union

from torch.utils.data import Dataset, IterableDataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry

@registry.register_dataset("normal")
class LongRopeDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and preprocessing long-text data.

    param data_path (str): Path to the data file.
    param tokenizer (Tokenizer): Tokenizer used to preprocess the data.
    param max_len (int): Maximum length of the sequence.
    param max_src_len (int): Maximum length of the input sequence.
    param mode (str, optional): Mode of the dataset, either 'pretrain' or 'sft'. Default is 'pretrain'.
    param read_nums (Union[int, None], optional): Number of samples to read from the data file, or None to read all. Default is None.
    param global_rank (int, optional): Global rank of the current process. Default is 0.
    param meta_prompt (str, optional): Meta prompt to be added to the input. Default is an empty string.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        meta_prompt:str ='',
        prefix:str='Q:',
        postfix:str='A:',
        *args,
        **kwargs
    ):
        self.build_data(
        data_path,
        tokenizer,
        max_len,
        max_src_len,
        mode,
        read_nums,
        global_rank,
        meta_prompt,
        prefix,
        postfix
    )
        self.proccess_data_file()
        
    def build_data(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        meta_prompt:str ='',
        prefix:str='Q:',
        postfix:str='A:'
    ):
        with open(data_path, "r", encoding="utf-8") as fh:
            if read_nums is None:
                read_nums = sum(1 for _ in fh)
        if read_nums > 10000:
            self.mininterval=10
        else:
            self.mininterval=0.1
        self.read_nums = read_nums
        self.all_data = []
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.meta_prompt = self.tokenizer.encode(meta_prompt, bos=True, eos=False)
        self.prefix = self.tokenizer.encode(prefix, bos=False, eos=False)
        self.postfix = self.tokenizer.encode(postfix, bos=False, eos=True)
        self.train_token_count = 0 
        self.global_rank = global_rank
        self.data_path = data_path
        print_rank_0(f'--->using dataset: {data_path}', global_rank)
        print_rank_0(f'--->training mode: {mode}', global_rank)
        print_rank_0(f'--->tokenizer name: {type(tokenizer).__name__}', global_rank)
        print_rank_0(f'--->using meta prompt: {meta_prompt}, prefix: {prefix}, postfix: {postfix}', global_rank)

    def proccess_data_file(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            with tqdm(fh, total=self.read_nums, desc='loading the dataset', disable=(self.global_rank != 0), mininterval=self.mininterval) as tbar:
                    for i, line in enumerate(tbar):
                        if i < self.read_nums:
                            try:
                                sample = json.loads(line.strip())
                            except:
                                sample = line.strip()
                                if i==0:
                                    print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
                            input_ids, output_ids = self.process_sample(sample)
                            postfix={"train_tokens":self._get_post_fix(self.train_token_count)}   
                            tbar.set_postfix(postfix)
                            self.all_data.append({"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(output_ids)})
                        else:
                            break
        print_rank_0(f'--->train_tokens:{self._get_post_fix(self.train_token_count)}', self.global_rank)
        
    def process_sample(self, sample):
        """
        Preprocesses a single data sample.

        param sample (dict): A dictionary containing the input and output sequences.

        Returns:
        input_ids (list): The preprocessed input sequence.
        output_ids (list): The preprocessed output sequence.
        """
        if isinstance(sample, dict) and "input_ids" in sample.keys():
            if isinstance(sample["input_ids"], str):
                input_ids = eval(sample["input_ids"])
            else:
                input_ids = sample["input_ids"]
            input_ids = self.meta_prompt + self.prefix + input_ids + self.postfix
            self.train_token_count += len(input_ids)
            output_ids = []
        else:
            if self.mode == 'sft':
                # In the case of sft, the input sample mast be a instance of dict.
                input_text = sample["input"] 
                output_text = sample["output"]
                input_ids = self.meta_prompt + self.prefix + self.tokenizer.encode(input_text, bos=False, eos=False) + self.postfix
                output_ids = self.tokenizer.encode(output_text, eos=True)
                self.train_token_count += len(output_ids)
            else:
                # In the case of pretrain, the input sample can be a single string.
                if isinstance(sample, dict):
                    assert "input" in sample.keys(), "Can not find input information in the dataset"
                    input_ids = self.meta_prompt + self.prefix + self.tokenizer.encode(sample["input"], bos=False, eos=False) + self.postfix
                    input_ids += self.tokenizer.encode(sample["output"], eos=True) if "output" in sample.keys() else []
                elif isinstance(sample, str):
                    input_ids = self.meta_prompt + self.prefix + self.tokenizer.encode(sample, bos=False, eos=False) + self.postfix
                else:
                    raise ValueError("You are using a not supported file format, please use jsonl or txt.")
                self.train_token_count += len(input_ids)
                output_ids = []

        if len(input_ids) > self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            output_ids = output_ids[:(self.max_len - len(input_ids))]
        input_len = len(input_ids)
        input_ids = input_ids + output_ids

        if self.mode == 'sft':
            labels = [self.tokenizer.pad_id] * input_len + output_ids
        elif self.mode == 'pretrain':
            labels = input_ids
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [self.tokenizer.pad_id] * pad_len

        assert len(input_ids) == len(labels)
        assert len(input_ids) <= self.max_len
        return input_ids, labels

    def _get_post_fix(self, count):
        if count >= 1e9:
            postfix = f"{count / 1e9:.2f}b"
        elif count >= 1e6:
            postfix = f"{count / 1e6:.2f}m"
        elif count >= 1e3:
            postfix = f"{count / 1e3:.2f}k"
        else:
            postfix = count
        return postfix

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        return self.all_data[idx]
    
@registry.register_dataset('iterable')
class IterableDataset(IterableDataset, LongRopeDataset):
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        meta_prompt:str ='',
        prefix:str='Q:',
        postfix:str='A:',
        shuffle:bool=False,
        *args,
        **kwargs
    ):
        LongRopeDataset.build_data(
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
        postfix
    )
        self.shuffle = shuffle
        
    def _load_samle(self, i, line):
        try:
            sample = json.loads(line.strip())
        except:
            sample = line.strip()
            if i==0:
                print_rank_0('--->Failed to load jsonl file, check if you use the correct format.', self.global_rank)
        return sample
    
    def __iter__(self):
        if self.shuffle:
            self._shuffle_iter()
        with open(self.data_path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i < self.read_nums:
                    sample = self._load_samle(i, line)
                    input_ids, output_ids = LongRopeDataset.process_sample(self, sample)
                    yield {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(output_ids)}
                else:
                    return StopIteration()
                
    def _shuffle_iter(self):
        read_indices = list(range(self.read_nums))
        random.shuffle(read_indices)
        with open(self.data_path, "r", encoding="utf-8") as fh:
            for i, read_idx in enumerate(read_indices):
                    for j, line in enumerate(fh):
                        if j==read_idx:
                            sample = self._load_samle(i, line)
                            input_ids, output_ids = LongRopeDataset.process_sample(self, sample)
                            yield {"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(output_ids)}
        return StopIteration()
                
    def __len__(self):       
        return self.read_nums
                
@registry.register_dataset('base')
class BaseDataSet(Dataset):
    def __init__(self):
        pass
    def __getitem__(self):
        pass
    def __len__(self):
        pass

if __name__ == '__main__':
    # Test example.
    file_path = '/home/bingxing2/ailab/scx6mh7/workspace/data/ruozhiba_qa.jsonl'
    tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/gemma/tokenizer.model'
    tokenizer = BaseTokenizer(tokenizer_path)

    iterable_dataset = IterableDataset(file_path,
                                       tokenizer,
                                       1500,
                                       1500,
                                       prefix='',
                                       postfix='',
                                       shuffle=True)
    for i, data in enumerate(iterable_dataset):
        print(i, data)