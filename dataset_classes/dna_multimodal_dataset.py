import re
import torch
from typing import Union, Optional

from dataset_classes import BaseDataset, DatasetConfig, BaseIterableDataset
from model.tokenizer import BaseTokenizer
from common.registry import registry

@registry.register_dataset('multimodal_dna_dataset')
class MultimodalDNADataSet(BaseDataset):
    def __init__(        
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        dataset_config: DatasetConfig,
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        multimodal_tokenizer = None,
        *args,
        **kwargs):
        
        BaseDataset.build_data(
        self,
        data_path,
        tokenizer,
        dataset_config,
        read_nums,
        global_rank
        )
        self.dna_tokenizer = multimodal_tokenizer
        self.project_token_num = kwargs.get('multimodal_k_tokens', 32)
        self._process_data_file()
        
    def process_sample(self, sample):
        input_ids = []
        dna_ids = []
        dna_ids_indicater = []
        pos = 0
        pattern = r'[ACTG]{6,}'
        first_text_piece_tag = True

        input_text, output_text = self._extract_texts(sample)
        self._process_text(input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag)
        if self.mode == 'sft':
            output_ids = self._encode_text(output_text)
        else:
            if output_text:
                input_ids.append(self._encode_text(output_text))
            self.train_token_count += len(input_ids)

        if self.mode == 'pretrain':
            input_ids.append(self.tokenizer.eos_id)
        else:
            output_ids.append(self.tokenizer.eos_id)

        if len(input_ids) > self.max_src_len:
            print(f'--->Length of source data excceed at rank {self.global_rank}: required length: {len(input_ids)} while max source length: {self.max_src_len}, cuttfing off')
            input_ids = input_ids[:self.max_src_len]
        if len(output_ids) > (self.max_len - len(input_ids)):
            print(f'--->Length of entire data instance excceed at rank {self.global_rank}, required length: {len(output_ids) + len(input_ids)} while max source length: {self.max_len}, cuttfing off')
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
            labels = [self.pad_id] * input_len + output_ids
        elif self.mode == 'pretrain':
            labels = input_ids
        attention_masks = [1] * len(input_ids)
        if self.padding:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.pad_id] * pad_len
            labels += [self.pad_id] * pad_len
            attention_masks += [0] * pad_len

        assert len(input_ids) == len(labels) == len(attention_masks)
        assert len(input_ids) <= self.max_len
        return {"input_ids": torch.LongTensor(input_ids), 
                "dna_ids": torch.LongTensor(dna_ids),
                "labels": torch.LongTensor(labels),
                "attention_masks": torch.LongTensor(attention_masks),
                "before_dna": dna_ids_indicater,
                "cal_metric_pos": cal_metric_pos}

    def _process_text(self, input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag):
        # Currently, only one DNA sequence is supported, or will cause error.
        for match in re.finditer(pattern, input_text):
            start, end = match.span()
            if pos < start:
                if first_text_piece_tag:
                    word_ids = [self.tokenizer.bos_id] if self.tokenizer.bos_id else []
                    word_ids += self.meta_prompt + self.prefix + self._encode_text(input_text[pos:start])
                    first_text_piece_tag = False
                else:
                    word_ids = self._encode_text(input_text[pos:start]) 
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
        input_ids += self.postfix

@registry.register_dataset('iterable_multimodal_dna_dataset')
class IterableMultimodalDNADataSet(MultimodalDNADataSet, BaseIterableDataset):
    def __init__(        
        self,
        data_path: str,
        tokenizer: BaseTokenizer,
        dataset_config: DatasetConfig,
        read_nums: Union[int, None] = None,
        global_rank: int=0,
        shuffle: bool = False,
        num_dp_ranks: Optional[int] = None,
        dp_rank: Optional[int] = None,
        seed: int = 42,
        start_step: int = 0,
        multimodal_tokenizer = None,
        *args,
        **kwargs):
        
        BaseDataset.build_data(
        self,
        data_path,
        tokenizer,
        dataset_config,
        read_nums,
        global_rank
        )
        self.dna_tokenizer = multimodal_tokenizer
        self.project_token_num = kwargs.get('multimodal_k_tokens', 32)
        BaseIterableDataset.init_parallel_and_shuffle(self, shuffle, num_dp_ranks, dp_rank, read_nums, seed, start_step)

    def __iter__(self):
        with open(self.data_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # E.g. latest checkpoint is step_10000.ckpt.
        # And micro batch size used before is 4
        # Start step should be 4*10000 = 40000.
        # This time the 40001th data should be read.
        step = self._get_start_step()
        # Equals to for read_idx in self.read_indices:
        while step < len(self.read_indices):
            read_idx = self.read_indices[step]
            line = lines[read_idx]
            step += 1
            if line:
                sample = self._load_sample(read_idx, line)
                yield MultimodalDNADataSet.process_sample(self, sample)