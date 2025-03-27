"""
This file can merge to vocab file in one.
"""
import json
from sentencepiece import sentencepiece_model_pb2 as pb2model

from common.registry import registry

def main(args):
    def get_spm_tokens(tokenizer_name, vocab_file_path):
        if tokenizer_name:
            tokenizer = registry.get_tokenizer_class(tokenizer_name)
            spm = pb2model.ModelProto()
            spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())
            spm_tokens = set(p.piece for p in spm.pieces)
        elif vocab_file_path:
            spm_tokens = set(json.load(open(vocab_file_path))['model']['vocab'].keys())
        else:
            raise ValueError("Either 'tokenizer_name' or 'vocab_file_path' must be provided.")
        return spm_tokens

    main_spm_tokens = get_spm_tokens(args.main_tokenizer_name, args.main_vocab_file_path)
    sub_spm_tokens = get_spm_tokens(args.sub_tokenizer_name, args.sub__vocab_file_path)
    
    print(f'num of tokens in main vocab:{len(main_spm_tokens)}')
    print(f'num of tokens in sub vocab:{len(sub_spm_tokens)}')

    main_spm = pb2model.ModelProto()
    for token in main_spm_tokens:
        new_p = main_spm.pieces.add()
        new_p.piece = token
        new_p.score = 0
    
    for token in sub_spm_tokens:
        if token not in main_spm_tokens:
            new_p = main_spm.pieces.add()
            new_p.piece = token
            new_p.score = 0
            
    print(f'num of tokens in merged vocab: {len(main_spm.pieces)}')

    with open(args.save_spm_path, 'wb') as f:
        f.write(main_spm.SerializeToString())
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main-tokenizer-name", type=str, default=None)
    parser.add_argument("--sub-tokenizer--name", type=str, default=None)
    parser.add_argument("--main-vocab-file-path", type=str, default=None)
    parser.add_argument("--sub-vocab-file-path", type=str, default=None)
    parser.add_argument("--save-spm-path", type=str, required=True)

    args=parser.parse_args()
    main(args)