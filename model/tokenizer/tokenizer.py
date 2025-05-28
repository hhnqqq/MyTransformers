import re
import os
import warnings
import tiktoken
import itertools

from pathlib import Path
from functools import partial
from tiktoken.load import load_tiktoken_bpe
from sentencepiece import SentencePieceProcessor
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import List, Dict, Optional, Tuple, Sequence, Union, Literal, AbstractSet, Collection, cast, Iterator

from common.registry import registry
from common.utils import ignore_module_print


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
        if self.pad_id == -1:
            self.pad_id = 0 
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = True, eos: bool = False, encode_single_gene=False) -> List[int]:
        """Converts a string into a list of tokens."""
        assert isinstance(s, str)
        if encode_single_gene:
            t = []
            pos = 0
            pattern = r'[ACTG]{6,}'
            for match in re.finditer(pattern, s):
                start, end = match.span()
                if pos < start:
                    t.extend(self.sp_model.encode(s[pos:start]))
                t.extend([self.sp_model.encode(i)[0] for i in s[start:end]])
                pos = end
            if pos < len(s):
                t.extend(self.sp_model.encode(s[pos:]))
        else:
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
    
@registry.register_tokenizer("dna")
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

    def encode(self, dna_seq: str, n_words: int, overlap: bool = True) -> List[int]:
        """
        Encode a DNA sequence into a list of k-mer IDs.
        """
        dna_ids = []
        if overlap:
            kmer_range = range(len(dna_seq) - self.k + 1)
        else:
            kmer_range = range(0, len(dna_seq), self.k)

        if self.k == 1:
            dna_ids = [self.dna_vocab_dict[i] + (n_words) for i in dna_seq]
        for i in kmer_range:
            kmer = dna_seq[i:i + self.k]
            if kmer in self.dna_vocab:
                dna_ids.append(self.dna_vocab_dict[kmer] + (n_words))
            else:
                dna_ids.append(self.dna_vocab_dict[kmer] + (n_words))  # Unknown token ID
        return dna_ids

    def decode(self, dna_ids: List[int]) -> str:
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

@registry.register_tokenizer("nl_dna")
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
                    result_ids += DNATokenizer.encode(self, sub_str, self.n_words)
                else:
                    result_ids += self.sp_model.encode(sub_str) 
            if bos:
                result_ids = [self.bos_id] + result_ids
            if eos:
                result_ids = result_ids + [self.eos_id]
            return result_ids
        else:
            return BaseTokenizer.encode(self, s, bos, eos)
        
@registry.register_tokenizer("llama3")
class Llama3Tokenizer:
    # TODO: 添加单核苷酸分词功能
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.

    For special tokens:
        A typical usage is: tokens.append(self.tokenizer.special_tokens["<|begin_of_text|>"])
        If direct encode is needed ---> please set search_special = True in encode method.
    """

    special_tokens: Dict[str, int]

    num_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        assert os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        num_base_tokens = len(mergeable_ranks)
        special_tokens = [
            "<|begin_of_text|>", # bos id
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>", # dna
            "<|reserved_special_token_2|>", # rna
            "<|reserved_special_token_3|>", # protein
            "<|start_header_id|>", # start of role
            "<|end_header_id|>", # end of role
            "<|reserved_special_token_4|>", 
            "<|eot_id|>",  # end of turn
        ] + [
            f"<|reserved_special_token_{i}|>"
            for i in range(5, self.num_reserved_special_tokens - 5)
        ]
        self.special_tokens = {
            token: num_base_tokens + i for i, token in enumerate(special_tokens)
        }
        self.special_token_pattern = r'\<\|.*?\|\>'
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_words: int = self.model.n_vocab
        # BOS / EOS token IDs
        self.bos_id: int = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: int = self.special_tokens["<|end_of_text|>"]
        # self.pad_id: int = -1
        self.pad_id: int = self.special_tokens["<|reserved_special_token_0|>"]
        self.stop_tokens = {
            self.special_tokens["<|end_of_text|>"],
            self.special_tokens["<|eot_id|>"],
        }
        self.eot_id = self.special_tokens["<|eot_id|>"]

    def encode(
        self,
        s: str,
        *,
        bos: bool = True,
        eos: bool = False,
        encode_single_gene = False,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
        search_special: bool = True
    ) -> List[int]:
        assert type(s) is str
        encode_func = partial(self.model.encode,
                              allowed_special=allowed_special,
                              disallowed_special=disallowed_special)
        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException.
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consecutive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        substrs = (
            substr
            for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            for substr in self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        """
        substrs = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            sub_string = s[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
            substr_list = self._split_whitespaces_or_nonwhitespaces(sub_string, MAX_NO_WHITESPACES_CHARS)
            substrs.extend(substr_list)"""
        t: List[int] = []
        for substr in substrs:
            if search_special:
                matches = list(re.finditer(self.special_token_pattern, substr))
                if matches:
                    current_idx = 0
                    for match in matches:
                        if match.group() in self.special_tokens:
                            encode_result = [] if current_idx == match.start() else encode_func(substr[current_idx:match.start()])
                            t.extend(encode_result + [self.special_tokens[match.group()]])
                        else:
                            t.extend(encode_func(substr[current_idx:match.end()]))
                        current_idx = match.end()
                    if current_idx < len(substr):
                        t.extend(encode_func(substr[current_idx:]))
                else:
                    t.extend(encode_func(substr))
            else:
                t.extend(encode_func(substr))
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, t: Sequence[int]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        return self.model.decode(cast(List[int], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: int
    ) -> Iterator[str]:
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() if len(s) > 0 else False
        slice_start = 0

        for i in range(len(s)):
            # 是否是空白字符
            is_now_space = s[i].isspace()

            if current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            else:
                # 如果不是空白字符
                current_slice_len += 1
                if current_slice_len > max_consecutive_slice_len:
                    yield s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        yield s[slice_start:]


@registry.register_tokenizer('dna_hyena')
class HyenaDNATokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids"]

    def __init__(self,
                 model_max_length: int=1000002,
                 bos_token="[BOS]",
                 eos_token="[SEP]",
                 sep_token="[SEP]",
                 cls_token="[CLS]",
                 pad_token="[PAD]",
                 mask_token="[MASK]",
                 unk_token="[UNK]",
                 **kwargs):
        """Character tokenizer for Hugging Face transformers.
        Args:
            characters (Sequence[str]): List of desired characters. Any character which
                is not included in this list will be replaced by a special token called
                [UNK] with id=6. Following are list of all of the special tokens with
                their corresponding ids:
                    "[CLS]": 0
                    "[SEP]": 1
                    "[BOS]": 2
                    "[MASK]": 3
                    "[PAD]": 4
                    "[RESERVED]": 5
                    "[UNK]": 6
                an id (starting at 7) will be assigned to each character.
            model_max_length (int): Model maximum sequence length.
        """
        self.characters = ('A', 'C', 'G', 'T', 'N')
        self.model_max_length = model_max_length

        self._vocab_str_to_int = {
            "[CLS]": 0,
            "[SEP]": 1,
            "[BOS]": 2,
            "[MASK]": 3,
            "[PAD]": 4,
            "[RESERVED]": 5,
            "[UNK]": 6,
            **{ch: i + 7 for i, ch in enumerate(self.characters)},
        }
        self._vocab_int_to_str = {v: k for k, v in self._vocab_str_to_int.items()}
        add_prefix_space = kwargs.pop("add_prefix_space", False)
        padding_side = kwargs.pop("padding_side", "left")

        super().__init__(
            bos_token=bos_token,
            eos_token=eos_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            unk_token=unk_token,
            add_prefix_space=add_prefix_space,
            model_max_length=model_max_length,
            padding_side=padding_side,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return len(self._vocab_str_to_int)

    def _tokenize(self, text: str) -> List[str]:
        return list(text)

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab_str_to_int.get(token, self._vocab_str_to_int["[UNK]"])

    def _convert_id_to_token(self, index: int) -> str:
        return self._vocab_int_to_str[index]

    def convert_tokens_to_string(self, tokens):
        return "".join(tokens)

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )

        result = ([0] * len(token_ids_0)) + [1]
        if token_ids_1 is not None:
            result += ([0] * len(token_ids_1)) + [1]
        return result

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        # cls = [self.cls_token_id]
        result = token_ids_0 + sep
        if token_ids_1 is not None:
            result += token_ids_1 + sep
        return result

    def get_vocab(self) -> Dict[str, int]:
        return self._vocab_str_to_int

    # HyenaDNA has a fixed vocabulary with no vocab file
    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple:
        return ()

@registry.register_tokenizer("dnabert2")
class DnaBert2Tokenizer:
    def __init__(self, model_path: Optional[str]):
        with warnings.catch_warnings(), ignore_module_print():
            warnings.simplefilter("ignore")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
   
        # BOS / EOS token IDs
        self.n_words: int = len(self.tokenizer.vocab)
        self.bos_id: int = self.tokenizer.bos_token_id
        self.eos_id: int = self.tokenizer.eos_token_id
        self.pad_id: int = self.tokenizer.pad_token_id

        if self.pad_id is None:
            self.pad_id = 0

    def encode(self, s: str, bos: bool=False, eos: bool=False) -> List[int]:
        assert self.is_valid_sequence(s), 'The input string is not fully composed by dna sequence'
        t = self.tokenizer.encode(s, add_special_tokens=False)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.tokenizer.decode(t)

    def vocab_size(self):
        return self.n_words

    def get_piece_size(self):
        return self.n_words
    
    @staticmethod
    def is_valid_sequence(seq):
        return set(seq.upper()) <= set('ATCG')

if __name__ == '__main__':
    dna_str = "Our objective is to accurately identify core promoter regions within human DNA sequences, focusing on the central region closest to the transcription start site (TSS) and start codon. A shorter context window is provided. Please carefully identify the core promoter region for the following DNA sequence delimited by special markers: %dna%TACTAATTGGGGCTCCGCATCTTCCAGTTACCTCATGCATGGCAGAGACTTTCTGGCGGGGAGGAGGAGG%dna%"
    llama_tokenizer = BaseTokenizer('/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model')
    print(llama_tokenizer.encode(dna_str, encode_single_gene=True))