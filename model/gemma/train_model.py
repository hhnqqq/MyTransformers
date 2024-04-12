import torch
import torch.nn.functional as F
from model.base_model import BaseModel
from torch.utils.checkpoint import checkpoint
from model.gemma.model import GemmaForCausalLM, precompute_freqs_cis
from common.registry import registry

@registry.register_model('gemma')
class TrainModel(BaseModel):
    """
    Trainer class for Gemma, responsible for handling input and output during training.
    """
    def __init__(self, model:GemmaForCausalLM, args, pad_id):
        """
        Initializes basic attributes for the trainer class and precomputes fixed values.

        param model: Gemma model with pretrained weight.
        param args: Arguments from argument parser.
        param pad_id: Pad index of the tokenizer, used to set ignore index for loss function.
        """
        super().__init__(args, pad_id)
        self.model = model.model
        self.embedder = model.embedder
        self.emb_weight = model.embedder.weight
        self.attention_mask = TrainModel.get_masks(args.max_len)
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                         args.max_len,
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi)
    
    def forward(self, input_ids, labels):
        return super().forward(input_ids, labels)
    
    def embedding(self, input_ids):
        hidden_states = F.embedding(input_ids, self.emb_weight)
        hidden_states = hidden_states * (torch.tensor(self.args.hidden_size)**0.5)
        attention_mask = self.attention_mask.to(hidden_states.device).to(hidden_states.dtype)
        return hidden_states, attention_mask
    
    def model_forward(self, hidden_states, freqs_cis, attention_mask):
        # Using activation checkpoint to reduce memory consumption or not.
        if self.args.activation_checkpoint:
            logits = checkpoint(self.model, hidden_states, freqs_cis, attention_mask, self.args.atten_type)
        else:
            logits = self.model(hidden_states=hidden_states, freqs_cis=freqs_cis, mask=attention_mask, atten_type=self.args.atten_type)
        logits = torch.matmul(logits, self.emb_weight.t().to(hidden_states.device).to(hidden_states.dtype))
        return logits

    @staticmethod
    def get_masks(seq_len, device='cpu', dtype=torch.float):
        attention_mask = torch.full((1, 1, seq_len, seq_len),
                    -2.3819763e38).to(torch.float)
        attention_mask = torch.triu(attention_mask, diagonal=1).to(device=device).to(dtype=dtype)
        return attention_mask