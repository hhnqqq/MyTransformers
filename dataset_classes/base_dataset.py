# @Author: haonan he
import json
import torch
from tqdm import tqdm
from typing import Optional

from torch.utils.data import Dataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry

@registry.register_dataset("normal")
class BaseDataset(Dataset):
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
        read_nums: Optional[int] = None,
        global_rank: int=0,
        meta_prompt: str ='',
        prefix: str='Q:',
        postfix: str='A:',
        cal_metric_pos: Optional[int] = None,
        encode_single_gene: bool = False,
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
        postfix,
        cal_metric_pos,
        encode_single_gene
    )
        self.process_data_file()
        
    def build_data(
        self,
        data_path: str,
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
        encode_single_gene: bool = False
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
        self.has_meta_prompt = meta_prompt and prefix and postfix
        self.has_prefix_postfix = prefix and postfix
        if self.has_meta_prompt:
            # eos id is only use for end the output of model.
            self.meta_prompt = self.tokenizer.encode(meta_prompt, bos=True, eos=False)
            self.prefix = self.tokenizer.encode(prefix, bos=False, eos=False)
            self.postfix = self.tokenizer.encode(postfix, bos=False, eos=False)
        elif self.has_prefix_postfix:
            self.prefix = self.tokenizer.encode(prefix, bos=True, eos=False)
            self.postfix = self.tokenizer.encode(postfix, bos=False, eos=False) 
        else:
            self.meta_prompt = self.prefix = self.postfix = []
        self.train_token_count = 0 
        self.global_rank = global_rank
        self.data_path = data_path
        self.cal_metric_pos = cal_metric_pos
        self.encode_single_gene = encode_single_gene

        print_rank_0(f'--->using dataset: {data_path}', global_rank)
        print_rank_0(f'--->training mode: {mode}', global_rank)
        print_rank_0(f'--->tokenizer name: {type(tokenizer).__name__}', global_rank)
        print_rank_0(f'--->using meta prompt: {meta_prompt}, prefix: {prefix}, postfix: {postfix}', global_rank)

    def process_data_file(self):
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
                            self.all_data.append(self.process_sample(sample))
                            postfix={"train_tokens":self._get_post_fix(self.train_token_count)}   
                            tbar.set_postfix(postfix)
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
        input_ids = []
        if isinstance(sample, dict) and "input_ids" in sample.keys():
            # In case input ids has been provided in the data file.
            if isinstance(sample["input_ids"], str):
                input_ids = eval(sample["input_ids"])
            else:
                input_ids = sample["input_ids"]
            self.train_token_count += len(input_ids)
            output_ids = []
        else:
            """
            Be careful for the use of eos_id and bos_id. 
            e.g. for Llama3.1-instruct: 
            single turn:
                input: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nuser_input
                output: <|start_header_id|>assistant<|end_header_id|>\n\nmodel_output<|end_of_text|>
            multi turns:
                input: <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nuser_input
                output: <|start_header_id|>assistant<|end_header_id|>\n\nmodel_output<|eot_id|>           
                input: <|start_header_id|>user<|end_header_id|>\n\nuser_input
                output: <|start_header_id|>assistant<|end_header_id|>\n\nmodel_output<|eot_id|>
                <eot_id> stands for end of a turn

            In this code base:
                set: 
                prefix=<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n and use bos to encode
                postfix=<|start_header_id|>assistant<|end_header_id|>\n\n

                and only use eos in output ids
            """
            input_text, output_text = self._extract_texts(sample)
            self._process_text(input_text, input_ids)
            if self.mode == 'sft':
                # In the case of sft, the input sample must be a instance of dict.
                output_ids = self.tokenizer.encode(output_text, bos=False, eos=True, encode_single_gene=self.encode_single_gene)
                self.train_token_count += len(output_ids)
            else:
                # In the case of pretrain, the input sample can be a single string.:
                input_ids += self.tokenizer.encode(output_text, bos=False, eos=True, encode_single_gene=self.encode_single_gene) if "output" in sample.keys() else []
                self.train_token_count += len(input_ids)
                output_ids = []

        if len(input_ids) > self.max_src_len:
            print_rank_0(f'--->Length of source data excceed: required length: {len(input_ids)} while max source length: {self.max_src_len}, cuttfing off', self.global_rank)
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            print_rank_0(f'--->Length of entire data instance excceed, cuttfing off', self.global_rank)
            output_ids = output_ids[:(self.max_len - len(input_ids))]
        input_len = len(input_ids)
        output_len = len(output_ids)
        input_ids = input_ids + output_ids
        if self.cal_metric_pos is not None:
            # 1 stand for eos token 
            cal_metric_pos = input_len + 1 + self.cal_metric_pos
        elif output_len == 3:
            cal_metric_pos = input_len + 1 
        else:
            cal_metric_pos = None

        if self.mode == 'sft':
            labels = [self.tokenizer.pad_id] * input_len + output_ids
        elif self.mode == 'pretrain':
            labels = input_ids
        pad_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
        labels = labels + [self.tokenizer.pad_id] * pad_len

        assert len(input_ids) == len(labels)
        assert len(input_ids) <= self.max_len
        return {"input_ids": torch.LongTensor(input_ids), 
                "labels": torch.LongTensor(labels),
                "cal_metric_pos": cal_metric_pos}

    def _extract_texts(self, sample):
        if self.mode == 'sft':
            return sample["input"], sample["output"]
        else:
            if isinstance(sample, dict):
                assert "input" in sample.keys(), "Can not find input information in the dataset"
                input_text = sample["input"]
                output_text = sample["output"] if "output" in sample.keys() else None
            elif isinstance(sample, str):
                input_text = sample
                output_text = None
            else:
                raise ValueError("You are using a not supported file format, please use jsonl or txt.")
            return input_text, output_text

    def _process_text(self, input_text, input_ids):
        input_ids += (self.tokenizer.encode(input_text, bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                        if self.has_prefix_postfix else 
                        self.tokenizer.encode(input_text, bos=True, eos=False, encode_single_gene=self.encode_single_gene))
        input_ids = self.meta_prompt + self.prefix + input_ids + self.postfix
        
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

