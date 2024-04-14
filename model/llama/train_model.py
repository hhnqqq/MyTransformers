import torch
from torch.utils.checkpoint import checkpoint

from common.registry import registry
from model.base_model import BaseModel
from model.llama.model import Transformer, precompute_freqs_cis

@registry.register_train_model('llama')
class LLaMaTrainModel(BaseModel):
    """
    Trainer class for llama, responsible for handling input and output during training.
    """
    def __init__(self, model:Transformer, args, pad_id):
        """
        Initializes basic attributes for the trainer class and precomputes fixed values.

        param model: llama model with pretrained weight.
        param args: Arguments from argument parser.
        param pad_id: Pad index of the tokenizer, used to set ignore index for loss function.
        """
        super().__init__(args, pad_id)
        self.layers = model.layers
        self.embedder = model.tok_embeddings
        self.output = model.output
        self.norm = model.norm
        self.attention_mask = LLaMaTrainModel.get_masks(args.max_len)
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                         args.max_len,
                                         theta=args.rope_theta,
                                         train_pi=args.train_pi,
                                         train_pipeline=False)
        
    def forward(self, input_ids, labels):
        return super().forward(input_ids, labels)
    
    def embedding(self, input_ids):
        hidden_states = self.embedder(input_ids)
        attention_mask = self.attention_mask.to(hidden_states.device).to(hidden_states.dtype)
        return hidden_states, attention_mask
    
    def model_forward(self, hidden_states, freqs_cis, attention_mask):
        # Using activation checkpoint to reduce memory consumption or not.
        for i in range(self.args.num_layers):
            if self.args.activation_checkpoint:
                    logits = checkpoint(self.layers[i], 
                                        hidden_states, 
                                        0, 
                                        freqs_cis, 
                                        attention_mask, 
                                        self.args.atten_type)
            else:
                logits = self.layers[i](hidden_states=hidden_states, 
                                        start_pos=0, 
                                        freqs_cis=freqs_cis, 
                                        mask=attention_mask, 
                                        atten_type=self.args.atten_type)
        logits = self.norm(logits)
        logits = self.output(logits)
        return logits
    
    
    @staticmethod
    def get_masks(seqlen, device='cpu', dtype=torch.float, start_pos=0):
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device),mask]).to(dtype)
            return mask