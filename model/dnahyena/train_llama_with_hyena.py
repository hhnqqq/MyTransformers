import re
import torch
import torch.nn as nn

from typing import List, Union, Tuple, Optional

from common.registry import registry
from model.projector import get_multimodal_projector
from model.llama.model import precompute_freqs_cis, Transformer
from model.dnahyena.hyena_model import HyenaDNAModel, _init_weights
from model import HyenaDNATokenizer, BaseTokenizer, Llama3Tokenizer


@registry.register_model(["llama1_with_hyena", "llama2_with_hyena"])
class LlamaWithHyena(nn.Module):
    def __init__(self, config, tokenizer:Optional[str]=None):
        super().__init__()
        self.config = config
        hyena_config = config.multimodal_model_config
        self.model = Transformer(config)
        self.multimodal_model = HyenaDNAModel(hyena_config)
        self.multimodal_tokenizer = HyenaDNATokenizer(model_max_length=hyena_config.layer_config.l_max)
        self.multimodal_model.multimodal_projector = get_multimodal_projector(config)
        _init_weights(self.multimodal_model.multimodal_projector, hyena_config)
        try:
            if tokenizer is None:
                self.tokenizer = BaseTokenizer(config.tokenizer)
            else:
                self.tokenizer = BaseTokenizer(tokenizer)
        except:
            pass
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2,
            config.rope_theta
        )

    @torch.inference_mode()
    def generate(
        self,
        inputs: Union[List[str], str],
        device,
        output_len: int,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        assert hasattr(self, "tokenizer"), "Can not inference with out a provieded tokenizer"
        if isinstance(inputs, str):
            inputs = [inputs]
        bsz = len(inputs)
            
        input_embs = []    
        for input_sentence in inputs:
            t = []
            pos = 0
            pattern = r'[ACTG]{6,}'
            for match in re.finditer(pattern, input_sentence):
                start, end = match.span()
                if pos < start:
                    word_ids = torch.LongTensor(self.tokenizer.encode(input_sentence[pos:start])).to(device)
                    word_embs = self.model.tok_embeddings(word_ids)
                    t.append(word_embs)
                dna_ids = torch.LongTensor(self.multimodal_tokenizer.encode(input_sentence[start:end])).to(device)
                dna_embs = self.multimodal_model(dna_ids.unsqueeze(0))
                dna_embs = self.multimodal_model.multimodal_projector(dna_embs)
                t.append(dna_embs.squeeze())
                pos = end
            if pos < len(input_sentence):
                word_ids = torch.LongTensor(self.tokenizer.encode(input_sentence[pos:start])).to(device)
                word_embs = self.model.tok_embeddings(word_ids)
                t.append(word_embs)
            input_embs.append(torch.cat(t, dim=0))

        params = self.model.params

        min_prompt_len = min(t.shape[0] for t in input_embs)
        max_prompt_len = max(t.shape[0] for t in input_embs)
        total_len = output_len + max_prompt_len

        pad_id = self.model.tok_embeddings(torch.LongTensor([self.tokenizer.pad_id]).to(device)).squeeze()
        pad_input_embs = pad_id.unsqueeze(0).unsqueeze(0).expand(bsz, total_len, -1).clone()
        for i, input_emb in enumerate(input_embs):
            pad_input_embs[i, : len(input_emb)] = input_emb

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=device)
        input_text_mask = (pad_input_embs != pad_id)[:, :, 0]

        # remove kv cache from attention attrs, for unified API
        caches_kv = []
        for _ in range(params.num_hidden_layers):
            n_kv_heads = params.n_kv_heads if params.n_kv_heads else params.n_heads
            size = (bsz, params.max_seq_len, n_kv_heads,
                    params.head_dim)
            dtype = params.get_dtype()
            cache_k = torch.zeros(size=size, dtype=dtype, device=device)
            cache_v = torch.zeros(size=size, dtype=dtype, device=device)
            caches_kv.append((cache_k, cache_v))

        out_tokens = [[] for i in range(bsz)]
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens=pad_input_embs, 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis,
                                        atten_type='',
                                        caches_kv=caches_kv,
                                        is_embed=True)
        for cur_pos in range(min_prompt_len, total_len):
            logits = self.model.forward(tokens=pad_input_embs[:, prev_pos:cur_pos], 
                                        start_pos=prev_pos, 
                                        freqs_cis=self.freqs_cis, 
                                        atten_type='',
                                        caches_kv=caches_kv,
                                        is_embed=True)
            next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)

            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            for i, token in enumerate(next_token.tolist()):
                out_tokens[i].append(token)
            pad_input_embs[:, cur_pos] = self.model.tok_embeddings(next_token.to(torch.long))
            prev_pos = cur_pos
            if all(eos_reached):
                break

        out_words = []
        print(out_tokens)
        for i, sample in enumerate(out_tokens):
            # cut to max gen len
            if self.tokenizer.eos_id in sample:
                eos_idx = sample.index(self.tokenizer.eos_id)
                sample = sample[:eos_idx]
            # out_tokens.append(toks)
            out_words.append(self.tokenizer.decode(sample))
        return out_words
        

@registry.register_model("llama3_with_hyena")
class Llama3WithHyena(LlamaWithHyena):
    def __init__(self, config, tokenizer:Optional[str]=None):
        super().__init__()
        if tokenizer is None:
            self.tokenizer = Llama3Tokenizer(config.tokenizer)
        else:
            self.tokenizer = Llama3Tokenizer(tokenizer)

if __name__ == '__main__':
    model_config = registry.get_model_config_class("llama1_with_hyena_large")()
    model_config.tokenizer = '/home/bingxing2/ailab/scx6mh7/workspace/llama/llama1_tokenizer.model'
    model_config.hyena_config.device = 'cuda'
    test_model = LlamaWithHyena(model_config)
    test_model.to(torch.float16)
    test_model.to('cuda')
    print(test_model.generate('this is a test: ATCGATCGATCG', 
                   device=torch.device('cuda'),
                   output_len=10))