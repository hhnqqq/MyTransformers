import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from deepspeed.pipe import LayerSpec, PipelineModule

from common.registry import registry
from model.gemma.model import precompute_freqs_cis
from model.gemma.model import GemmaForCausalLM
from model.gemma.train_model import GemmaTrainModel

class EmbeddingPipelineLayer(nn.Module):
    def __init__(self, model: GemmaForCausalLM, args):
        super().__init__()
        self.args = args
        self.embedder = model.embedder
        self.weight = self.embedder.weight
        # if args.quant:
        #     self.weight_scaler = self.word_embeddings.weight_scaler

    def forward(self, inputs):
        # attention mask和input还需要处理一下, [batch_size, input_len, 1]
        input_ids, labels = inputs
        # 经过embedder计算, [batch_size, input_len, hidden_size]
        hidden_states = F.embedding(input_ids, self.weight)
        # gemma要使用hidden size对embedding输出进行正则化
        hidden_states = hidden_states * (torch.tensor(self.args.hidden_size)**0.5)
        # 获得attention mask, 这里还需要验证一下
        attention_mask = GemmaTrainModel.get_masks(input_ids.shape[1], device=hidden_states.device, dtype=hidden_states.dtype)
        # 获得rope频率
        freqs_cis = precompute_freqs_cis(self.args.head_dim,
                                         input_ids.shape[1],
                                         theta=self.args.rope_theta,
                                         train_pi=self.args.train_pi,
                                         train_pipeline=True).to(hidden_states.device)
        freqs_cis.requires_grad_(True)
        attention_mask.requires_grad_(True)
        return hidden_states, freqs_cis, attention_mask, labels

class DecoderPipelineLayer(nn.Module):
    # 还需要对k,v cache进行处理
    def __init__(self, model: GemmaForCausalLM, layer_idx, args):
        super().__init__()
        self.layer = model.model.layers[layer_idx]
        self.args = args

    def forward(self, inputs):
        hidden_states, freqs_cis, attention_mask, labels = inputs
        # [batch_size, input_len, hidden_dim]
        if self.args.activation_checkpoint:
            hidden_states = checkpoint(self.layer, hidden_states, freqs_cis, attention_mask, self.args.atten_type)
        else:
            hidden_states = self.layer(hidden_states, freqs_cis, attention_mask, self.args.atten_type)
        return hidden_states, freqs_cis, attention_mask, labels
    
class FNormPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM):
        super().__init__()
        self.final_norm = model.model.norm
        self.emb_weight = model.embedder.weight.t()

    def forward(self, inputs):
        hidden_states, _, _, labels = inputs
        # [batch_size, input_len, hidden_dim]
        logits = self.final_norm(hidden_states)
        logits = torch.matmul(logits, self.emb_weight.to(hidden_states.device).to(hidden_states.dtype))
        return logits, labels
    
class SamplerPipelineLayer(torch.nn.Module):
    def __init__(self, model: GemmaForCausalLM):
        super().__init__()
        self.embedder = model.embedder
        self.weight = self.embedder.weight

    def forward(self, inputs):
        hidden_states, labels = inputs
        # [batch_size, input_len, vocab_size]
        logits = torch.matmul(hidden_states, self.weight.t().to(hidden_states.device).to(hidden_states.dtype))
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

@registry.register_pipeline_model("gemma")
def get_pipeline_model(model, args):
    layers = [LayerSpec(EmbeddingPipelineLayer, model=model, args=args),
            *[LayerSpec(DecoderPipelineLayer, model=model, args=args, layer_idx=idx) for idx in
            range(args.num_layers)],
            LayerSpec(FNormPipelineLayer, model=model),
            LayerSpec(LossPipelineLayer, pad_id=args.pad_id)]
    return PipelineModule(layers=layers, num_stages=args.num_pp_stages, partition_method='uniform')