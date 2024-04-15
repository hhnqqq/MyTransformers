# @Author: haonan he
# @modified by: zhijian jiang
import json
import torch
from tqdm import tqdm
from typing import Union

from torch.utils.data import Dataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0

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
        meta_prompt:str =''
    ):
        self.all_data = []
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_src_len = max_src_len
        self.meta_prompt = meta_prompt
        self.train_token_count = 0 
        print_rank_0(f'--->using dataset: {data_path}', global_rank)
        print_rank_0(f'--->training mode: {self.mode}', global_rank)
        print_rank_0(f'--->meta prompt: {self.meta_prompt}', global_rank)
        print_rank_0(f'--->tokenizer name: {type(self.tokenizer).__name__}', global_rank)
        with open(data_path, "r", encoding="utf-8") as fh:
            if read_nums is None:
                read_nums = sum(1 for _ in fh)
            with open(data_path, "r", encoding="utf-8") as fh:
                with tqdm(fh, total=read_nums, desc='loading the dataset', disable=(global_rank != 0)) as tbar:
                        for i, line in enumerate(tbar):
                            if i < read_nums:
                                sample = json.loads(line.strip())
                                input_ids, output_ids = self.preprocess_sample(sample)
                                postfix={"train_tokens":self._get_post_fix(self.train_token_count)}   
                                tbar.set_postfix(postfix)
                                self.all_data.append({"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(output_ids)})
        print_rank_0(f'--->train_tokens:{self._get_post_fix(self.train_token_count)}', global_rank)

    def preprocess_sample(self, sample):
        """
        Preprocesses a single data sample.

        param sample (dict): A dictionary containing the input and output sequences.

        Returns:
        input_ids (list): The preprocessed input sequence.
        output_ids (list): The preprocessed output sequence.
        """
        if self.mode == 'sft':
            # In the case of sft, the input sample mast be a instance of dict.
            input_text = self.meta_prompt + 'Q:' + sample["input"] + '\n'
            output_text = 'A:' + sample["output"]
            input_ids = self.tokenizer.encode(input_text)
            output_ids = self.tokenizer.encode(output_text)
            self.train_token_count += len(output_ids)
        else:
            # In the case of pretrain, the input sample can be a single string.
            if isinstance(sample, dict):
                assert "input" in sample.keys(), "Can not find input information in the dataset"
                input = sample["input"] + sample["output"] if "output" in sample.keys() else sample["input"]
            elif isinstance(sample, str):
                input = sample
            else:
                raise ValueError("You are using a not supported file format, please use jsonl or txt.")
            input_ids = self.tokenizer.encode(input)
            self.train_token_count += len(input_ids)
            output_ids = []

        if len(input_ids) > self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            output_ids = output_ids[:(self.max_len - len(input_ids))]
        input_ids = input_ids + output_ids

        if self.mode == 'sft':
            labels = [self.tokenizer.pad_id] * len(input_ids) + output_ids[len(input_ids):]
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
    
if __name__ == '__main__':
    # Test example.
    z = LongRopeDataset('/home/modelfun/zhaokangkang/mini_LLama/gemma-data/LongQLoRA-SFT-Data-2k.jsonl', 
                        Tokenizer('/home/modelfun/zhaokangkang/mini_LLama/gemma/tokenizer.model'), 1500, 256, 'pretrain',1000,1)
    print(z[0])