import os
import torch
import torch.nn as nn
from model.tokenizer import BaseTokenizer
from sentencepiece import sentencepiece_model_pb2 as pb2model

def initialize_new_embeddings(existing_embeddings, num_new_tokens, method='normal', **kwargs):
    """
    initialize new embedding
    
    Args:
        existing_embeddings: pretrained embeddings which not require re-initialize
        num_new_tokens: number of new tokens
        method: initialze method
    returns: 
        new embeddings after initialize
    """
    existing_weight = existing_embeddings.weight.data
    embedding_dim = existing_weight.size(1)
    
    if method == 'normal':
        mean = torch.mean(existing_weight)
        std = torch.std(existing_weight)
        return torch.normal(mean=mean, std=std, size=(num_new_tokens, embedding_dim))
    
    elif method == 'uniform':
        low = existing_weight.min()
        high = existing_weight.max()
        return torch.rand(num_new_tokens, embedding_dim) * (high - low) + low
    
    elif method == 'xavier':
        return torch.nn.init.xavier_uniform_(torch.empty(num_new_tokens, embedding_dim))
    
    elif method == 'kaiming':
        return torch.nn.init.kaiming_normal_(torch.empty(num_new_tokens, embedding_dim))
    
    elif method == 'sub_words':
        tokenizer = kwargs.get('tokenizer')
        sub_vocab = kwargs.get('sub_vocab')
        new_embeddings = torch.zeros(num_new_tokens, embedding_dim)
        for i, word in enumerate(sub_vocab):
            sub_words = torch.LongTensor(tokenizer.encode(word, eos=False, bos=False))
            sub_embeddings = torch.stack([existing_embeddings(sub_word) for sub_word in sub_words])
            new_embeddings[i] = torch.mean(sub_embeddings, dim=0)
        return new_embeddings
    
    else:
        raise ValueError(f"Unsupported initialization method: {method}")

def main(args):
    model_sd = torch.load(args.pretrained_ckpt_path, map_location='cpu')
    embedding_sd = {'weight':v for k,v in model_sd.items() if args.embeddings_layer_name in k}
    output_sd = {'weight':v for k,v in model_sd.items() if args.output_layer_name in k}

    tokenizer = BaseTokenizer(args.base_tokenizer_path)
    expanded_tokenizer = BaseTokenizer(args.expanded_tokenizer_path)
    vocab_size = tokenizer.n_words
    expanded_vocab_size = expanded_tokenizer.n_words
    sub_vocab_size = expanded_vocab_size - vocab_size

    spm = pb2model.ModelProto()
    spm.ParseFromString(expanded_tokenizer.sp_model.serialized_model_proto())
    spm_tokens = [p.piece for p in spm.pieces]
    sub_vocab = spm_tokens[vocab_size:]

    assert sub_vocab_size == len(sub_vocab)

    embeddings = nn.Embedding(vocab_size, args.hidden_size)
    expanded_embeddings = nn.Embedding(expanded_vocab_size, args.hidden_size)
    output = nn.Linear(args.hidden_size, vocab_size, bias=False)
    expanded_output = nn.Linear(args.hidden_size, expanded_vocab_size, bias=False)

    embeddings.load_state_dict(embedding_sd)
    output.load_state_dict(output_sd)

    expanded_embeddings.weight.data[:vocab_size, :] = embeddings.weight.data
    expanded_output.weight.data[:vocab_size, :] = output.weight.data

    # initialize new embedding
    new_embeddings_weight = initialize_new_embeddings(
        embeddings, 
        sub_vocab_size, 
        method=args.init_method, 
        tokenizer=tokenizer, 
        sub_vocab=sub_vocab
    )
    
    expanded_embeddings.weight.data[vocab_size:, :] = new_embeddings_weight
    expanded_output.weight.data[vocab_size:, :] = new_embeddings_weight 

    print("Embedding stds and means:")
    print(f"Original: {torch.std(expanded_embeddings.weight.data[:vocab_size, :])}, {torch.mean(expanded_embeddings.weight.data[:vocab_size, :])}")
    print(f"New: {torch.std(expanded_embeddings.weight.data[vocab_size:, :])}, {torch.mean(expanded_embeddings.weight.data[vocab_size:, :])}")
    print("Output stds and means:")
    print(f"Original: {torch.std(expanded_output.weight.data[:vocab_size, :])}, {torch.mean(expanded_output.weight.data[:vocab_size, :])}")
    print(f"New: {torch.std(expanded_output.weight.data[vocab_size:, :])}, {torch.mean(expanded_output.weight.data[vocab_size:, :])}")

    save_sd = {
        '.'.join([args.embeddings_layer_name, 'weight']): expanded_embeddings.weight.data,
        '.'.join([args.output_layer_name, 'weight']): expanded_output.weight.data
    }

    torch.save(save_sd, os.path.join(args.save_dir, f'merged_embedding_{args.init_method}.ckpt'))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained-ckpt-path', type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth')
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--embeddings-layer-name', type=str, default='tok_embeddings')
    parser.add_argument('--output-layer-name', type=str, default='output')
    parser.add_argument('--base-tokenizer-path', type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model')
    parser.add_argument('--expanded-tokenizer-path', type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/dnabert2/merged_tokenizer.model')
    parser.add_argument('--init-method', type=str, default='normal', choices=['normal', 'uniform', 'xavier', 'kaiming', 'sub_words'])
    parser.add_argument('--save-dir', type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/dnabert2')
    args = parser.parse_args()

    main(args)