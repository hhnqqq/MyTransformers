import torch.nn as nn

from typing import Optional

from common.registry import registry
from model.projector import get_multimodal_projector
from model.llama.model import precompute_freqs_cis, Transformer
from model.dnabert.bert_model import BertModel
from model import BaseTokenizer, DnaBert2Tokenizer, Llama3Tokenizer


def initialize_transformer(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

@registry.register_model(["llama1_with_bert", "llama2_with_bert"])
class LlamaWithBert(nn.Module):
    def __init__(self, config, tokenizer:Optional[str]=None, multimodal_tokenizer:Optional[str]=None):
        super().__init__()
        self.config = config
        bert_config = config.multimodal_model_config
        self.model = Transformer(config)
        self.multimodal_model = BertModel(bert_config)
        self.multimodal_projector = get_multimodal_projector(config)
        initialize_transformer(self.multimodal_projector)
        try:
            if tokenizer is None:
                self.tokenizer = BaseTokenizer(config.tokenizer)
            else:
                self.tokenizer = BaseTokenizer(tokenizer)
            if multimodal_tokenizer is None:
                self.multimodal_tokenizer = DnaBert2Tokenizer(registry.get_path("tokenizer_dnabert2"))
            else:
                self.multimodal_tokenizer = DnaBert2Tokenizer(multimodal_tokenizer)
        except:
            pass
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads, 
            config.max_seq_len * 2,
            config.rope_theta
        )

    def forward(self):
        pass

@registry.register_model("llama3_with_bert")
class Llama3WithBert(LlamaWithBert):
    def __init__(self, config, tokenizer:Optional[str]=None):
        super().__init__(config)
        if tokenizer is None:
            self.tokenizer = Llama3Tokenizer(config.tokenizer)
        else:
            self.tokenizer = Llama3Tokenizer(tokenizer)