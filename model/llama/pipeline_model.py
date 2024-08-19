import torch
from torch.utils.checkpoint import checkpoint
from deepspeed.pipe import LayerSpec, PipelineModule

from common.registry import registry
from model.llama.model import LlamaGenerate
from model.llama.train_model import LLaMaTrainModel
from model.llama.model import precompute_freqs_cis

class EmbeddingPipelineLayer(torch.nn.Module):
    def __init__(self, model: LlamaGenerate, args):
        super().__init__()
        self.args = args
        self.embedder = model.model.tok_embeddings
        self.freqs_cis = precompute_freqs_cis(args.head_dim,
                                            args.max_len,
                                            theta=args.rope_theta,
                                            train_pi=args.train_pi,
                                            train_pipeline=True)
        # if args.quant:
        #     self.weight_scaler = self.word_embeddings.weight_scaler

    def forward(self, inputs):
        # [batch_size, input_len, 1]
        input_ids, labels = inputs
        # [batch_size, input_len, hidden_size]
        hidden_states = self.embedder(input_ids)
        # Acquire attention mask.
        attention_mask = LLaMaTrainModel.get_masks(input_ids.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
        freqs_cis = self.freqs_cis.to(hidden_states.device)
        # Have to set freqs_cis and attention mask trainable, or deepspeed will throw a exception.
        freqs_cis.requires_grad_(True)
        attention_mask.requires_grad_(True)
        return hidden_states, freqs_cis, attention_mask, labels
    
class DecoderPipelineLayer(torch.nn.Module):
    def __init__(self, model: LlamaGenerate, layer_idx, args):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.args = args

    def forward(self, inputs):
        hidden_states, freqs_cis, attention_mask, labels = inputs
        # [batch_size, input_len, hidden_dim]
        if self.args.activation_checkpoint:
            hidden_states = checkpoint(self.layer,  
                                       hidden_states, 
                                       0, 
                                       freqs_cis, 
                                       attention_mask, 
                                       self.args.atten_type, 
                                       use_reentrant=False)
        else:
            hidden_states = self.layer(hidden_states, 
                                       0, 
                                       freqs_cis, 
                                       attention_mask, 
                                       self.args.atten_type)
        return hidden_states, freqs_cis, attention_mask, labels
    
class FNormPipelineLayer(torch.nn.Module):
    def __init__(self, model: LlamaGenerate):
        super().__init__()
        self.final_norm = model.model.norm
        self.o_proj = model.model.output

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        # [batch_size, input_len, hidden_dim]
        logits = self.final_norm(hidden_states)
        logits = self.o_proj(logits)
        return logits, labels

class LossPipelineLayer(torch.nn.Module):
    def __init__(self, pad_id):
        super().__init__()
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, inputs):
        logits, labels = inputs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
        return loss

@registry.register_pipeline_model(["llama", "llama1", "llama2", "llama3"])
def get_pipeline_model(model, args):
    layers = [LayerSpec(EmbeddingPipelineLayer, model=model, args=args),
            *[LayerSpec(DecoderPipelineLayer, model=model, args=args, layer_idx=idx) for idx in
            range(args.num_layers)],
            LayerSpec(FNormPipelineLayer, model=model),
            LayerSpec(LossPipelineLayer, pad_id=args.pad_id)]
    return PipelineModule(layers=layers, num_stages=args.num_pp_stages, partition_method='uniform')