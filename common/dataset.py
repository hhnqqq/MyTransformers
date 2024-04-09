import json
import torch
from torch.utils.data import Dataset
from gemma.tokenizer import Tokenizer
from tqdm import tqdm
from typing import Union

class LongRopeDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len, max_src_len, mode:str='pretrain', read_nums:Union[int, None]=None):
        self.all_data = []
        with open(data_path, "r", encoding="utf-8") as fh:
            if read_nums is None:
                read_nums = sum(1 for _ in fh)
            for i, line in tqdm(enumerate(fh), total=read_nums, desc='start loading the dataset'):
                if i<read_nums:
                    sample = json.loads(line.strip())
                    meta_prompt = 'You are a helpful long context assitant: \n'
                    input_text = meta_prompt + 'Q:' + sample["input"] + '\n'
                    output_text = 'A:' + sample["output"]
                    # Tokenize input and output texts.
                    input_tokens = tokenizer.tokenize(input_text)
                    output_tokens = tokenizer.tokenize(output_text)

                    # Truncate input and output if they exceed maximum lengths.
                    if len(input_tokens) > max_src_len:
                        input_tokens = input_tokens[:max_src_len - 3]

                    if len(output_tokens) > (max_len - len(input_tokens) - 3):
                        output_tokens = output_tokens[:(max_len - len(input_tokens) - 3)]  # Adjust if needed.

                    # Combine tokens with special tokens.
                    tokens = input_tokens + output_tokens
                    input_ids = tokenizer.convert_tokens_to_ids(tokens)
                    # Pad input_ids with pad token for labels.
                    if mode == 'sft':
                        labels = [tokenizer.pad_id] * len(input_tokens) + input_ids[len(input_tokens):]
                    elif mode == 'pretrain':
                        labels = input_ids
                    pad_len = max_len - len(input_ids)
                    input_ids = input_ids + [tokenizer.pad_id] * pad_len
                    labels = labels + [tokenizer.pad_id] * pad_len

                    assert len(input_ids) == len(labels)
                    assert len(input_ids) <= max_len

                    self.all_data.append({"input_ids": torch.LongTensor(input_ids), "labels": torch.LongTensor(labels)})

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        instance = self.all_data[idx]
        return instance

if __name__ == '__main__':
    # Test example.
    pass