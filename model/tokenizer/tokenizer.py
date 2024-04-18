import re
import itertools
import os
import numpy as np
from typing import List, Optional
from common.registry import registry

from sentencepiece import SentencePieceProcessor


@registry.register_tokenizer("base")
class BaseTokenizer:

    def __init__(self, model_path: Optional[str]):
        # Reload tokenizer.
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs.
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """Converts a list of tokens into a string."""
        return self.sp_model.decode(t)

    def tokenize(self, text):
        return self.sp_model.EncodeAsPieces(text)

    def convert_tokens_to_string(self, tokens):
        return self.sp_model.DecodePieces(tokens)

    def convert_tokens_to_ids(self, tokens):
        return [self.sp_model.PieceToId(token) for token in tokens]

    def convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def convert_id_to_token(self, idx):
        return self.sp.IdToPiece(idx)

    def __len__(self):
        return self.num_tokens
    
class DNATokenizer():
    def __init__(self, k: int = 4):
        self.k = k
        self.dna_vocab = self.build_dna_vocab()
        self.dna_vocab_size = len(self.dna_vocab)
        self.dna_vocab_dict = {kmer: i for i, kmer in enumerate(self.dna_vocab)}
    
    def build_dna_vocab(self) -> List[str]:
        """
        Build a vocabulary of unique k-mers from the DNA dataset.
        """
        # Implement your logic to build the DNA vocabulary here
        dna_vocab = [''.join(x) for x in itertools.product('ACGT', repeat=self.k)]
        return dna_vocab

    def encode_dna(self, dna_seq: str, n_words: int, overlap: bool = True) -> List[int]:
        """
        Encode a DNA sequence into a list of k-mer IDs.
        """
        dna_ids = []
        if overlap:
            kmer_range = range(len(dna_seq) - self.k + 1)
        else:
            kmer_range = range(0, len(dna_seq), self.k)

        for i in kmer_range:
            kmer = dna_seq[i:i + self.k]
            if kmer in self.dna_vocab:
                dna_ids.append(self.dna_vocab_dict[kmer] + (n_words))
            else:
                dna_ids.append(self.dna_vocab_dict[kmer] + (n_words))  # Unknown token ID
        return dna_ids

    def decode_dna(self, dna_ids: List[int]) -> str:
        """
        Decode a list of k-mer IDs into a DNA sequence.
        """
        dna_seq = []
        for dna_id in dna_ids:
            if dna_id < self.dna_vocab_size:
                dna_seq.append(self.dna_vocab[dna_id])
            else:
                dna_seq.append('N')  # Unknown nucleotide
        return ''.join(dna_seq)

@registry.register_tokenizer("gemma_dna")
class GemmaDNATokenizer(DNATokenizer, BaseTokenizer):
    def __init__(self, model_path: str, k: int = 6):
        DNATokenizer.__init__(self, k)
        BaseTokenizer.__init__(self, model_path)
        self.new_vocab_size = self.n_words + self.dna_vocab_size
    
    def encode(self, s: str, bos: bool=True, eos: bool=False) -> List[int]:
        if '<dna>' in s and '</dna>' in s:
            result_ids = []
            pattern = r'<dna>([ACTG]*?)</dna>'
            split_groups = re.split(pattern, s)
            for sub_str in split_groups:
                if re.match(r'[ACTG]+', sub_str):
                    result_ids += self.encode_dna(sub_str, self.n_words)
                else:
                    result_ids += self.sp_model.encode(sub_str) 
            if bos:
                result_ids = [self.bos_id] + result_ids
            if eos:
                result_ids = result_ids + [self.eos_id]
            return result_ids
        else:
            return super().encode(s, bos, eos)
        
