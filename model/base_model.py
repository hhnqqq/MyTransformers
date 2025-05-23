import torch
import torch.nn as nn
import common.utils.parallel_states as parallel_states

from common.utils import cal_metric
from liger_kernel.transformers import LigerCrossEntropyLoss, LigerFusedLinearCrossEntropyLoss

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pad_id = args.pad_id
        self.fuse_linear_loss = self.args.fuse_linear_loss
        if args.loss_fct == 'mse':
            self.loss_fct = torch.nn.MSELoss()
        elif args.loss_fct == 'ce':
            if self.fuse_linear_loss:
                self.loss_fct = LigerFusedLinearCrossEntropyLoss(reduction="mean",
                                                                 ignore_index=self.pad_id)
            else:
                self.loss_fct = LigerCrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, **kwargs):
        input_ids, labels = kwargs["input_ids"], kwargs["labels"]
        input_ids, labels, freqs_cis = self.cut_sequence(input_ids, labels)
        hidden_states, attention_mask = self.embedding(input_ids)
        loss, logits = self.model_forward(hidden_states, labels, freqs_cis, attention_mask)
        if logits is not None:
            metrics = self.compute_metric(logits, labels, kwargs["cal_metric_pos_tensor"])
        else:
            metrics = {}
        return loss, metrics
    
    def cut_sequence(self, input_ids, labels=None):
        seq_parallel_world_size = parallel_states.get_sequence_parallel_world_size()
        seq_parallel_world_rank = parallel_states.get_sequence_parallel_rank()
        if self.args.atten_type is not None and 'ulysses' in self.args.atten_type:
            assert self.args.max_len % seq_parallel_world_size == 0, 'Max input length is not divisble by sequence parallel stages.'
            assert self.args.head_num % seq_parallel_world_size == 0, 'Attention head num is not divisble by sequence parallel stages.'
            # Split the input ids and lables and freqs cis for deepspeed-ulysses.
            seq_len_per_group = self.args.max_len // seq_parallel_world_size
            local_seq_start = seq_parallel_world_rank * seq_len_per_group
            local_seq_end = (seq_parallel_world_rank +1) * seq_len_per_group
            input_ids = input_ids[:, local_seq_start:local_seq_end]
            labels = labels[:, local_seq_start:local_seq_end] if labels else labels
            freqs_cis = self.freqs_cis[local_seq_start:local_seq_end,:].to(input_ids.device)
        else:
            freqs_cis = self.freqs_cis.to(input_ids.device)
        freqs_cis.requires_grad_(True)
        return input_ids, labels, freqs_cis
    
    def embedding(self, input_ids):
        raise NotImplementedError()
    
    def model_forward(self, hidden_states, labels, freqs_cis, attention_mask):
        raise NotImplementedError()
    
    def compute_loss(self, logits, labels, lm_head_weight=None):
        if self.fuse_linear_loss and lm_head_weight is None:
            raise ValueError('`lm_head_weight` can not be None when `fuse_linear_loss` is True.')
        
        shift_logits = logits[..., :-1, :].contiguous().reshape(-1, logits.size(-1))
        shift_labels = labels[..., 1:].contiguous().reshape(-1)
        if self.fuse_linear_loss:
            loss = self.loss_fct(lm_head_weight, shift_logits, shift_labels)
        else:
            loss = self.loss_fct(shift_logits, shift_labels)
            
        return loss
    
    def compute_metric(self, 
                       logits: torch.Tensor, 
                       labels: torch.Tensor, 
                       cal_metric_pos_tensor: torch.Tensor):
        if cal_metric_pos_tensor is None:
            return {}
        else:
            # gather后的形状和index tensor相同
            _, _, vocab_size = logits.shape
            target_logits_pos_tensor = (cal_metric_pos_tensor-1).view(-1, 1, 1).expand(-1, 1, vocab_size) # [bsz, 1, vocab_size]
            target_labels_pos_tensor = cal_metric_pos_tensor.view(-1, 1) # [bsz, 1]
            target_logits = torch.gather(logits, 1, target_logits_pos_tensor)
            target_logits = torch.argmax(target_logits, dim=-1) # [bsz, 1]
            target_labels = torch.gather(labels, 1, target_labels_pos_tensor) # [bsz, 1]
            return cal_metric(target_labels.cpu().numpy(), target_logits.cpu().numpy())

def prepare_4d_attention_mask(attention_mask_with_indices: "torch.Tensor", dtype: "torch.dtype") -> "torch.Tensor":
    r"""
    Expands the attention mask with indices from (batch_size, seq_len) to (batch_size, 1, seq_len, seq_len),
    while handles packed sequences and transforms the mask to lower triangular form to prevent future peeking.

    e.g.
    ```
    [[1, 1, 2, 2, 2, 0]]
    ```
    ->
    ```
    [
        [
            [
                [o, x, x, x, x, x],
                [o, o, x, x, x, x],
                [x, x, o, x, x, x],
                [x, x, o, o, x, x],
                [x, x, o, o, o, x],
                [x, x, x, x, x, x],
            ]
        ]
    ]
    ```
    where `o` equals to `0.0`, `x` equals to `min_dtype`.
    """
    bsz, seq_len = attention_mask_with_indices.size()
    min_dtype = float('-inf')
    # [bs, seq_len]->[bs, 1, seq_len, seq_len]
    expanded_mask = attention_mask_with_indices[:, None, None, :].expand(bsz, 1, seq_len, seq_len)
    # e.g: [[1, 1, 2, 2, 2, 0]] -> [[1, 1, 1, 1, 1, 0]]
    padding_mask = torch.where(expanded_mask != 0, 1, 0)
    attention_mask_4d = torch.eq(expanded_mask, expanded_mask.transpose(-1, -2)).int() * padding_mask
    attention_mask_4d *= torch.tril(torch.ones((seq_len, seq_len), dtype=torch.long))
    attention_mask_4d = torch.where(attention_mask_4d != 0, torch.tensor(0, dtype=dtype), min_dtype)
    return attention_mask_4d