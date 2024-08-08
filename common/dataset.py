# @Author: haonan he
# @modified by: zhijian jiang
import re
import json
import math
import torch
import random
from tqdm import tqdm
from typing import Union, Optional

from torch.utils.data import Dataset, IterableDataset
from model.tokenizer import BaseTokenizer, HyenaDNATokenizer
from common.utils import print_rank_0, is_seed_set
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
        self.meta_prompt = self.tokenizer.encode(meta_prompt, bos=True, eos=False) if meta_prompt is not None else []
        self.prefix = self.tokenizer.encode(prefix, bos=False, eos=False) if prefix is not None else []
        self.postfix = self.tokenizer.encode(postfix, bos=False, eos=True) if postfix is not None else []
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
                input_ids = (self.tokenizer.encode(input_text, bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                             if self.meta_prompt != [] else 
                             self.tokenizer.encode(input_text, eos=True, bos=True, encode_single_gene=self.encode_single_gene))
                input_ids = self.meta_prompt + self.prefix + input_ids + self.postfix
                output_ids = self.tokenizer.encode(output_text, eos=True, encode_single_gene=self.encode_single_gene)
                self.train_token_count += len(output_ids)
            else:
                # In the case of pretrain, the input sample can be a single string.
                if isinstance(sample, dict):
                    assert "input" in sample.keys(), "Can not find input information in the dataset"
                    input_ids = (self.tokenizer.encode(sample["input"], bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                                 if self.meta_prompt != [] else 
                                 self.tokenizer.encode(sample["input"], eos=True, bos=True, encode_single_gene=self.encode_single_gene))
                    input_ids = self.meta_prompt + self.prefix + input_ids + self.postfix
                    input_ids += self.tokenizer.encode(sample["output"], eos=True, encode_single_gene=self.encode_single_gene) if "output" in sample.keys() else []
                elif isinstance(sample, str):
                    input_ids = (self.tokenizer.encode(sample, bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                                 if self.meta_prompt != [] else 
                                 self.tokenizer.encode(sample, eos=True, bos=False, encode_single_gene=self.encode_single_gene))
                    input_ids = self.meta_prompt + self.prefix + input_ids + self.postfix
                else:
                    raise ValueError("You are using a not supported file format, please use jsonl or txt.")
                self.train_token_count += len(input_ids)
                output_ids = []

        if len(input_ids) > self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
            print_rank_0(f'--->Length of source data excceed: required length: {len(input_ids)} while max source length: {self.max_src_len}, cuttfing off', self.global_rank)
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
            print_rank_0(f'--->global rank:{self.global_rank} read range [{self.start}:{self.end}]', 0)
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
                        yield LongRopeDataset.process_sample(self, sample)
                
    def _shuffle_iter(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        for read_idx in self.read_indices:
            line = lines[read_idx]
            if line:
                sample = self._load_sample(read_idx, line)
                yield LongRopeDataset.process_sample(self, sample)
                
    def __len__(self):       
        return self.read_nums
                
@registry.register_dataset('multimodal_dna_dataset')
class MultimodalDNADataSet(LongRopeDataset):
    def __init__(        
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        meta_prompt: str ='',
        prefix: str = 'Q:',
        postfix: str = 'A:',
        cal_metric_pos: Optional[int] = None,
        encode_single_gene: bool = False,
        multimodal_tokenizer = None,
        *args,
        **kwargs):
        
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
        postfix,
        cal_metric_pos,
        encode_single_gene
        )
        self.dna_tokenizer = multimodal_tokenizer
        self.project_token_num = kwargs.get('multimodal_k_tokens', 32)
        self.process_data_file()
        
    def process_sample(self, sample):
        input_ids = []
        dna_ids = []
        dna_ids_indicater = []
        pos = 0
        pattern = r'[ACTG]{6,}'
        first_text_piece_tag = True

        if self.mode == 'sft':
            input_text, output_text = self._extract_texts(sample)
            self._process_text(input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag)
            output_ids = self.tokenizer.encode(output_text, eos=True, encode_single_gene=self.encode_single_gene)
            input_ids += self.postfix
        else:
            input_text, output_text = self._extract_texts(sample)
            self._process_text(input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag)
            if output_text:
                input_ids.append(self.tokenizer.encode(output_text, eos=True, encode_single_gene=self.encode_single_gene))
            self.train_token_count += self.input_ids_count

        if len(input_ids) > self.max_src_len:
            print_rank_0(f'--->Length of source data excceed: required length: {len(input_ids)} while max source length: {self.max_src_len}, cuttfing off', self.global_rank)
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            print_rank_0(f'--->Length of data excceed, cuttfing off', self.global_rank)
            output_ids = output_ids[:(self.max_len - len(input_ids))]

        input_len = len(input_ids)
        output_len = len(output_ids)
        input_ids += output_ids
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
        input_ids += [self.tokenizer.pad_id] * pad_len
        labels += [self.tokenizer.pad_id] * pad_len

        assert len(input_ids) == len(labels)
        assert len(input_ids) <= self.max_len
        return {"input_ids": torch.LongTensor(input_ids), 
                "dna_ids": torch.LongTensor(dna_ids),
                "labels": torch.LongTensor(labels),
                "before_dna": dna_ids_indicater,
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

    def _process_text(self, input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag):
        # Currently, only one DNA sequence is supported, or will cause error.
        for match in re.finditer(pattern, input_text):
            start, end = match.span()
            if pos < start:
                if first_text_piece_tag:
                    word_ids = (self.tokenizer.encode(input_text[pos:start], bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                                if self.meta_prompt != [] else 
                                self.tokenizer.encode(input_text[pos:start], bos=True, eos=False, encode_single_gene=self.encode_single_gene))
                    word_ids = self.meta_prompt + self.prefix + word_ids
                    first_text_piece_tag = False
                else:
                    word_ids = self.tokenizer.encode(input_text[pos:start], bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                input_ids += word_ids

            dna_ids += self.dna_tokenizer.encode(input_text[start:end])
            
            word_ids = [self.tokenizer.pad_id] * self.project_token_num
            pos = end
            if dna_ids_indicater == []:
                dna_ids_indicater.append(len(input_ids))
            input_ids += word_ids

            if pos < len(input_text):
                word_ids = self.tokenizer.encode(input_text[pos:len(input_text)], bos=False, eos=True)
                input_ids += word_ids

@registry.register_dataset('iterable_multimodal_dna_dataset')
class IterableMultimodalDNADataSet(IterableDataset, MultimodalDNADataSet):
    def __init__(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        max_len: int,
        max_src_len: int,
        mode: str = 'pretrain',
        read_nums: Union[int, None] = None,
        global_rank: int = 0,
        meta_prompt: str = '',
        prefix: str = 'Q:',
        postfix: str = 'A:',
        shuffle: bool = False,
        num_dp_ranks: Optional[int] = None,
        dp_rank: Optional[int] = None,
        cal_metric_pos: Optional[int] = None,
        encode_single_gene: bool = False,
        multimodal_tokenizer = None,
        *args,
        **kwargs
    ):
        # Call the __init__ method of MultimodalDNADataSet
        MultimodalDNADataSet.build_data(
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
        )

        self.dna_tokenizer = multimodal_tokenizer
        self.project_token_num = kwargs.get('project_token_num', 32)
        self.input_ids_count = 0
        self.shuffle = shuffle
        
        if num_dp_ranks and dp_rank is not None:
            self.dp = True
            read_nums_per_rank = math.ceil(self.read_nums / num_dp_ranks)
            self.start = read_nums_per_rank * dp_rank
            self.end = min(read_nums_per_rank * (dp_rank + 1), self.read_nums)
            print_rank_0(f'--->global rank:{self.global_rank} read range [{self.start}:{self.end}]', 0)
        else:
            self.start = 0
            self.end = self.read_nums

        read_indices = list(range(self.read_nums))
        self.read_indices = read_indices[self.start:self.end]
        # Make sure random seed has been set
        if num_dp_ranks:
            assert is_seed_set()
        random.shuffle(self.read_indices)

    def _load_sample(self, i, line):
        try:
            sample = json.loads(line.strip())
        except:
            sample = line.strip()
            if i == 0:
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
                        yield self.process_sample(sample)

    def _shuffle_iter(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        for read_idx in self.read_indices:
            line = lines[read_idx]
            if line:
                sample = self._load_sample(read_idx, line)
                yield self.process_sample(sample)

    def __len__(self):
        return self.read_nums
    
class RepeatingLoader:

    def __init__(self, loader):
        """Wraps an iterator to allow for infinite iteration. This is especially useful
        for DataLoader types that we wish to automatically restart upon completion.

        Args:
            loader (iterator): The data loader to repeat.
        """
        self.loader = loader
        self.data_iter = iter(self.loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            print_rank_0("--->Start a new iteration for repeating loader.", 0)
            self.data_iter = iter(self.loader)
            batch = next(self.data_iter)
        return batch


if __name__ == '__main__':
    # Test example.
    import os
    from utils import DataCollator, Timer
    from torch.utils.data import DataLoader
    from deepspeed.utils import RepeatingLoader
    os.environ['NO_LOG_FILE'] = 'true'
    file_path = '/home/bingxing2/ailab/scx6mh7/workspace/data/dev_gue_except_covid.jsonl'
    tokenizer_path = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama1_tokenizer.model'
    tokenizer = BaseTokenizer(tokenizer_path)
    dna_tokenizer = HyenaDNATokenizer(100002)
    data_collator = DataCollator(tokenizer)

    iterable_dataset = IterableMultimodalDNADataSet(file_path,
                                       tokenizer,
                                       multimodal_tokenizer=dna_tokenizer,
                                       max_len=360,
                                       max_src_len=350,
                                       mode='sft',
                                       prefix=None,
                                       postfix=None,
                                       meta_prompt=None,
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
        # print(data['labels'])