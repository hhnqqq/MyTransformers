import torch
# from torch.utils.checkpoint import checkpoint
import deepspeed

from common.utils import parallel_states as parallel_states
from common.registry import registry
from model.base_model import BaseModel
from model.llama.model import LlamaGenerate, precompute_freqs_cis
from common.utils.torch_hooks import hook_backward_norm_fn, save_grad, ParameterUpdateHook

@registry.register_train_model(["llama", "llama1", "llama2", "llama3"])
class LLaMaTrainModel(BaseModel):
    """
    Trainer class for llama, responsible for handling input and output during training.
    """
    def __init__(self, model:LlamaGenerate, args):
        """
        Initializes basic attributes for the trainer class and precomputes fixed values.

        param model: llama model with pretrained weight.
        param args: Arguments from argument parser.
        """
        super().__init__(args)
        self.layers = model.model.layers
        self.tok_embeddings = model.model.tok_embeddings
        self.output = model.model.output
        self.norm = model.model.norm
        if 'flash' in args.atten_type:
            self.attention_mask = None
        else:
            self.attention_mask = LLaMaTrainModel.get_masks(args.max_len)
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                            args.max_len,
                                            theta=args.rope_theta,
                                            train_pi=args.train_pi,
                                            train_pipeline=False)
        
    def forward(self, **kwargs):
        return super().forward(**kwargs)
    
    def embedding(self, input_ids):
        hidden_states = self.tok_embeddings(input_ids)
        if self.attention_mask is not None:
            attention_mask = self.attention_mask.to(hidden_states.device).to(hidden_states.dtype)
        else:
            attention_mask = self.attention_mask
        return hidden_states, attention_mask
    
    def model_forward(self, logits, labels, freqs_cis, attention_mask):
        # Using activation checkpoint to reduce memory consumption or not.
        for i in range(self.args.num_layers):
            if self.args.activation_checkpoint:
                logits = deepspeed.checkpointing.checkpoint(self.layers[i], 
                                    logits, 
                                    0, 
                                    freqs_cis, 
                                    attention_mask, 
                                    self.args.atten_type)
            else:
                logits = self.layers[i](x=logits, 
                                        start_pos=0, 
                                        freqs_cis=freqs_cis, 
                                        mask=attention_mask, 
                                        atten_type=self.args.atten_type)
        logits = self.norm(logits)
        if self.fuse_linear_loss:
            loss = self.compute_loss(logits, labels, self.output.weight)
            logits = None
        else:
            logits = self.output(logits)
            loss = self.compute_loss(logits, labels)
        return loss, logits
    
    
    @staticmethod
    def get_masks(seqlen, device='cpu', dtype=torch.float, start_pos=0):
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=device),mask]).to(dtype)
            return mask


@registry.register_train_model(["llama1_with_hyena", "llama2_with_hyena", "llama3_with_hyena",
                                "llama1_with_bert", "llama2_with_bert", "llama3_with_bert"])
class MultimodalLlamaTrainModel(LLaMaTrainModel):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.encode_fp32 = args.multimodal_encode_fp32
        self.multimodal_model = model.multimodal_model
        self.multimodal_projector = model.multimodal_projector
        # self.layers[0].register_full_backward_hook(hook_backward_fn)
        self.multimodal_projector.register_full_backward_hook(hook_backward_norm_fn)
        # self.multimodal_model.register_full_backward_hook(hook_backward_norm_fn)
        # hook = ParameterUpdateHook('multimodal_projector', self.multimodal_projector.weight, print_every=10)
        # self.multimodal_projector.weight.register_hook(hook)

    def forward(self, **kwargs):
        """
        Performs the forward pass of MultimodalLlamaTrainModel during training.

        model_config:
            input_ids (torch.Tensor): Shape [batch_size, seq_len]. DNA part is padded with pad_id.
            labels (torch.Tensor): Shape [batch_size, seq_len].
            dna_ids (torch.Tensor): Shape [batch_size, hyena_l_output].
            dna_ids_indicaters (torch.Tensor): Shape [batch_size, dna_sequence_count]. Currently, dna_sequence_count is restricted to 1.

        Returns:
            tuple: (loss, empty_dict)

        Forward process:
        1. Compute hidden states from input_ids: [batch_size, seq_len, hidden_size]
        2. Process dna_ids through hyena model: [batch_size, hyena_l_output, hyena_hidden_size]
        3. Project hyena output to word embedding space: [batch_size, hyena_l_output, hidden_size]
        4. Replace DNA part in hidden states with projected result according the indicaters, the mask operation equals to:
            `start_positions = dna_ids_indicaters
            end_positions = start_positions + dna_token_num
            for i in range(batch_size):
                hidden_states[i, start_positions[i]:end_positions[i]] = dna_hidden_states[i]`
        5. Pass modified hidden states through Llama model
        6. Compute and return loss
        """

        input_ids = kwargs["input_ids"]
        dna_ids = kwargs["dna_ids"]
        labels = kwargs["labels"]
        before_dna = kwargs["before_dna"]

        hidden_states = self.tok_embeddings(input_ids)
        if self.encode_fp32:
            self.multimodal_model.to(torch.float32)
        dna_hidden_states = self.encoder_forward(dna_ids, hidden_states.dtype)
        # dna_hidden_states.register_hook(save_grad('dna'))
        if self.encode_fp32:
            dna_hidden_states.to(input_ids.dtype)

        dna_token_num = dna_hidden_states.shape[1]
        start_positions = before_dna
        end_positions = start_positions + dna_token_num

        batch_size, seq_len, hidden_dim = hidden_states.shape

        # [batch_size, seq_len]
        index = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(before_dna.device)
        mask = (index >= start_positions) & (index < end_positions)
        mask = mask.unsqueeze(-1).expand(-1, -1, hidden_dim)

        # Scatter dna_hidden_states into hidden_states
        hidden_states = hidden_states.masked_scatter(mask, dna_hidden_states)

        hidden_states, labels, freqs_cis = self.cut_sequence(hidden_states, labels)
        if self.attention_mask is not None:
            attention_mask = self.attention_mask.to(hidden_states.device, dtype=hidden_states.dtype)
        else:
            attention_mask = None
        loss, _ = self.model_forward(hidden_states, labels, freqs_cis, attention_mask)
        
        return loss, {}
    
    def encoder_forward(self, dna_ids, origin_dtype):
        dna_hidden_states = self.multimodal_model(dna_ids)
        dna_hidden_states = self.multimodal_projector(dna_hidden_states)
        # dna_hidden_states.register_hook(save_grad('dna'))
        return dna_hidden_states