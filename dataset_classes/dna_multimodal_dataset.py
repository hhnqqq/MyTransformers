import re
import torch
from typing import Union, Optional

from dataset_classes import BaseDataset
from model.tokenizer import BaseTokenizer
from common.utils import print_rank_0
from common.registry import registry

@registry.register_dataset('multimodal_dna_dataset')
class MultimodalDNADataSet(BaseDataset):
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
        padding: bool = True,
        apply_chat_template: bool = False,
        multimodal_tokenizer = None,
        *args,
        **kwargs):
        
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
        padding,
        apply_chat_template
        )
        self.dna_tokenizer = multimodal_tokenizer
        self.project_token_num = kwargs.get('multimodal_k_tokens', 32)
        self.process_data_file()
        
    def process_sample(self, sample):
        input_ids = []
        dna_ids = []
        dna_ids_indicater = []
        output_dis = []
        pos = 0
        pattern = r'[ACTG]{6,}'
        first_text_piece_tag = True

        input_text, output_text = self._extract_texts(sample)
        self._process_text(input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag)
        if self.mode == 'sft':
            output_ids = self.tokenizer.encode(output_text, bos=True, eos=True, encode_single_gene=self.encode_single_gene)
            input_ids += self.postfix
        else:
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

    def _process_text(self, input_text, input_ids, dna_ids, dna_ids_indicater, pos, pattern, first_text_piece_tag):
        # Currently, only one DNA sequence is supported, or will cause error.
        for match in re.finditer(pattern, input_text):
            start, end = match.span()
            if pos < start:
                if first_text_piece_tag:
                    word_ids = (self.tokenizer.encode(input_text[pos:start], bos=False, eos=False, encode_single_gene=self.encode_single_gene) 
                                if self.has_meta_prompt else 
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
