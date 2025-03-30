# @Author: haonan he
import json
import torch
import logging
from tqdm import tqdm
from typing import Optional, Union
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from torch.utils.data import Dataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry
from dataset_classes.dataset_tools import get_line_count

@dataclass
class DatasetConfig:
    # More feasible if any new parameter should be added.
    max_len: int
    max_src_len: int
    meta_prompt: str =''
    input_field: str = 'input'
    output_field: str = 'output'
    mode: str = 'pretrain',
    prefix: str='Q:'
    postfix: str='A:'
    padding: bool = True
    apply_chat_template: bool = False
    cal_metric_pos: Optional[int] = None
    encode_single_gene: bool = False

@registry.register_dataset("normal")
class BaseDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading and preprocessing long-text data.

    Args:
        data_path (str): Path to the data file.
        tokenizer (BaseTokenizer): Tokenizer used to preprocess the data.
        max_len (int): Maximum length of the sequence.
        max_src_len (int): Maximum length of the input sequence.
        mode (str, optional): Mode of the dataset, either 'pretrain' or 'sft'. Default is 'pretrain'.
        read_nums (Union[int, None], optional): Number of samples to read from the data file, or None to read all. Default is None.
        global_rank (int, optional): Global rank of the current process. Default is 0.
        meta_prompt (str, optional): Meta prompt to be added to the input. Default is an empty string.
        prefix (str, optional): Prefix to be added before the input sequence. Default is 'Q:'.
        postfix (str, optional): Postfix to be added after the input sequence. Default is 'A:'.
        cal_metric_pos (Optional[int], optional): Position for calculating metrics. Default is None.
        encode_single_gene (bool, optional): Whether to encode single genes. Default is False.
        padding (bool, optional): Whether to pad the sequences. Default is True.
        apply_chat_template (bool, optional): Whether to use `apply_chat_template` method for huggingface tokenizer. Default is True.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Union[BaseTokenizer, PreTrainedTokenizerBase],
        dataset_config: DatasetConfig,
        read_nums: Optional[int] = None,
        global_rank: int=0,
        *args,
        **kwargs
    ):
        self.build_data(
        data_path,
        tokenizer,
        dataset_config,
        read_nums,
        global_rank,
    )
        self.process_data_file()
        
    def build_data(
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        dataset_config: DatasetConfig,
        read_nums: Optional[int] = None,
        global_rank: int=0
    ):
    
        self.mininterval=0.1
        if isinstance(data_path ,str):
            line_count = get_line_count(data_path)
            if read_nums is None:
                    read_nums = line_count
            if read_nums > 10000:
                self.mininterval=10
        else:
            line_count = None
        self.line_count = line_count
        self.read_nums = read_nums
        self.all_data = []
        self.mode = dataset_config.mode
        self.tokenizer = tokenizer
        self.train_token_count = 0 
        self.global_rank = global_rank
        self.data_path = data_path
        
        self.input_field = dataset_config.input_field
        self.output_field = dataset_config.output_field
        self.cal_metric_pos = dataset_config.cal_metric_pos
        self.encode_single_gene = dataset_config.encode_single_gene
        self.padding = dataset_config.padding 
        self.apply_chat_template = dataset_config.apply_chat_template and isinstance(tokenizer, PreTrainedTokenizerBase)
        if self.apply_chat_template:
            self.meta_prompt = dataset_config.meta_prompt
            print_rank_0('--->Prefix and postfix will be ignored when `apply_chat_template` is true', global_rank)
        else:
            self.init_data_format(dataset_config.meta_prompt, 
                                  dataset_config.prefix, 
                                  dataset_config.postfix)
        if dataset_config.max_src_len > dataset_config.max_len:
            # To ensure the assertion error would not be triggered.
            self.max_len = dataset_config.max_src_len
            self.max_src_len = dataset_config.max_len
            print_rank_0(f'--->max_src_len is greater than max_len, swraped', global_rank)
        else:
            self.max_len = dataset_config.max_len
            self.max_src_len = dataset_config.max_src_len

        print_rank_0(f'--->using dataset: {data_path}', global_rank)
        print_rank_0(f'--->training mode: {self.mode}', global_rank)
        print_rank_0(f'--->tokenizer name: {type(tokenizer).__name__}', global_rank)
        print_rank_0(f'--->using meta prompt: {dataset_config.meta_prompt},' 
                     'prefix: {dataset_config.prefix},'
                     'postfix: {dataset_config.postfix}', 
                     global_rank)


    def init_data_format(self, meta_prompt, prefix, postfix):
        self.meta_prompt = self._process_data_format_encode(meta_prompt)
        self.prefix = self._process_data_format_encode(prefix)
        self.postfix = self._process_data_format_encode(postfix)

    def _process_data_format_encode(self, input_data):
        return self._encode_text(input_data) if input_data else []

    def process_data_file(self):
        # Enable a tqdm progress bar.
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

        Args:
            sample (dict): A dictionary containing the input and output sequences.

        Returns:
            input_ids (list): The preprocessed input sequence.
            output_ids (list): The preprocessed output sequence.
        """
        if self.apply_chat_template:
            # In case of HuggingFace tokenizer, we can use apply_chat_template for easy formatting.
            # This require the mode is sft and max_src_len will be ignored.
            input_text, output_text = self._extract_texts(sample)
            if not (self.mode == 'sft' and output_text):
                raise ValueError("apply_chat_template requires SFT mode and non-empty output text.")
            messages = [{'role':'system', 'content': self.meta_prompt}] if self.meta_prompt else []
            messages.extend([{'role':'user', 'content': input_text}, {'role':'assistant', 'content': output_text}])
            tokenized = self.tokenizer.apply_chat_template(
                        messages,
                        max_length=self.max_len,
                        return_assistant_tokens_mask=True,
                        truncation=True,
                        return_dict=True)
            input_ids, output_mask = tokenized['input_ids'], tokenized['assistant_masks']
            output_len = sum(output_mask) or len(input_ids)
            # If assistant can not be accessed, input_ids will also used to compute loss.
            input_len = len(input_ids) - output_len
            input_ids, output_ids = input_ids[:-output_len], input_ids[-output_len:]
            self.train_token_count += output_len if output_len else input_len
        else:
            input_text, output_text, input_ids, output_ids = self.preprocess_sample(sample)
            if output_text:
                if self.mode == 'sft':
                    # In the case of sft, the input sample must be a instance of dict.
                    output_ids = self._encode_text(output_text)
                    self.train_token_count += len(output_ids)
                else:
                    # In the case of pretrain, the input sample can be a single string.:
                    # Make sure that the output text can be tokenized.
                    input_ids += [] if output_text is None else self._encode_text(output_text) 
                    self.train_token_count += len(input_ids)

            if self.mode == 'pretrain':
                # Make sure that the last token id is eos id
                input_ids.append(self.tokenizer.eos_id)
            else:
                # In the case of sft, try to acquire eot id (Only exist in Llama3Tokenzier class)
                # output_ids.append(getattr(self.tokenizer, "eot_id", self.tokenizer.eos_id))
                output_ids.append(self.tokenizer.eos_id)
            if len(input_ids) > self.max_src_len:
                print(input_text)
                print(f'--->Length of source data excceed at rank {self.global_rank}: required length: {len(input_ids)} while max source length: {self.max_src_len}, cuttfing off')
                input_ids = input_ids[:self.max_src_len]
            if len(output_ids) > (self.max_len - len(input_ids)):
                print(f'--->Length of entire data instance excceed at rank {self.global_rank}, required length: {len(output_ids) + len(input_ids)} while max source length: {self.max_len}, cuttfing off')
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
            labels = [getattr(self.tokenizer, 'label_pad_id', self.tokenizer.pad_id)] * input_len + output_ids
        elif self.mode == 'pretrain':
            labels = input_ids
        attention_masks = [1] * len(input_ids)
        if self.padding:
            # Do not need to pad when stretegy is packing.
            pad_len = self.max_len - len(input_ids)
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [getattr(self.tokenizer, 'label_pad_id', self.tokenizer.pad_id)] * pad_len
            attention_masks += [0] * pad_len

        assert len(input_ids) == len(labels) == len(attention_masks)
        assert len(input_ids) <= self.max_len
        return {"input_ids": torch.LongTensor(input_ids), 
                "labels": torch.LongTensor(labels),
                "attention_masks": torch.LongTensor(attention_masks),
                "cal_metric_pos": cal_metric_pos}

    def preprocess_sample(self, sample):
        if isinstance(sample, dict) and "input_ids" in sample.keys():
            return self.preprocess_sample_from_ids(sample)
        else:
            """
            Be careful for the use of special tokens and input format. 
            Fine-tuning a chat model must exactly reproduced the official format.
            e.g. for Llama3.1-instruct: 
            <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{ system_prompt }}<|eot_id|>
            <|start_header_id|>user<|end_header_id|>\n\n{{ user_msg_1 }}<|eot_id|>
            <|start_header_id|>assistant<|end_header_id|>\n\n{{ model_answer_1 }}<|eot_id|>
            In this code base:
                set: 
                meta_prompt=<|start_header_id|>system<|end_header_id|>\n\n{{ system_prompt }}<|eot_id|>
                prefix=<|start_header_id|>user<|end_header_id|>\n\n
                postfix=<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

                The final input ids list will equals to [bos id] + meta_prompt + prefix + input + postfix + output + [eot id]
            """
            input_text, output_text = self._extract_texts(sample)
            input_ids = self._process_text(input_text)
            return input_text, output_text, input_ids, []
        
    def preprocess_sample_from_ids(self, sample):
        # In case input ids has been provided in the data file.
        if isinstance(sample["input_ids"], str):
            input_ids = eval(sample["input_ids"])
        else:
            input_ids = sample["input_ids"]
        self.train_token_count += len(input_ids)
        return '', '', input_ids, []
    
    def _extract_texts(self, sample):
        """
        Extracts input and output texts from the sample.

        Args:
            sample (Union[dict, str]): The data sample.

        Returns:
            tuple: A tuple containing input text and output text.
        """
        if self.mode == 'sft':
            if not isinstance(sample, dict):
                print_rank_0(r'--->Error! Traning mode is sft while sample is not a dict.', self.global_rank, logging.ERROR)
            return sample[self.input_field], sample[self.output_field]
        else:
            # This is competible with pretraining mode and reinforce learning mode.
            if isinstance(sample, dict):
                if not self.input_field in sample.keys():
                    raise NameError(f"Can not find {self.input_field} information in the dataset")
                input_text = sample[self.input_field]
                output_text = sample[self.output_field] if self.output_field in sample.keys() else None
            elif isinstance(sample, str):
                input_text = sample
                output_text = None
            else:
                raise ValueError("You are using a not supported file format, please use jsonl or txt.")
            return input_text, output_text

    def _process_text(self, input_text):
        """
        Processes and encodes the input text.

        Args:
            input_text (str): The input text to be processed.

        Returns:
            list: A list of token IDs.
        """
        encoded_ids = self._encode_text(input_text)
        # Make sure that the bos id always be the first id of input ids.
        # In some case (such as Qwen2.5-7b), these is no bos_id in tokenizer.
        input_ids = [self.tokenizer.bos_id] if self.tokenizer.bos_id else []
        input_ids += self.meta_prompt + self.prefix + encoded_ids + self.postfix
        return input_ids

    def _encode_text(self, text):
        if isinstance(self.tokenizer, PreTrainedTokenizerBase):
            return self.tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.tokenizer.encode(
                text, 
                bos=False, 
                eos=False, 
                encode_single_gene=self.encode_single_gene
            )

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
