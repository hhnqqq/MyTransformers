import torch
import torch.nn as nn
import common.utils.parallel_states as parallel_states

from common.utils import cal_metric

class BaseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pad_id = args.pad_id
        if args.loss_fct == 'mse':
            self.loss_fct = torch.nn.MSELoss()
        elif args.loss_fct == 'ce':
            self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, **kwargs):
        input_ids, labels = kwargs["input_ids"], kwargs["labels"]
        input_ids, labels, freqs_cis = self.cut_sequence(input_ids, labels)
        hidden_states, attention_mask = self.embedding(input_ids)
        logits = self.model_forward(hidden_states, freqs_cis, attention_mask)
        loss = self.compute_loss(logits, labels)
        return loss, self.compute_metric(logits, labels, kwargs["cal_metric_pos_tensor"])
    
    def cut_sequence(self, input_ids, labels):
        seq_parallel_world_size = parallel_states.get_sequence_parallel_world_size()
        seq_parallel_world_rank = parallel_states.get_sequence_parallel_rank()
        if self.args.atten_type is not None and 'ulysses' in self.args.atten_type:
            assert self.args.max_len % seq_parallel_world_size == 0, 'Max input length is not divisble by sequence parallel stages.'
            assert self.args.head_nums % seq_parallel_world_size == 0, 'Attention head num is not divisble by sequence parallel stages.'
            # Split the input ids and lables and freqs cis for deepspeed-ulysses.
            seq_len_per_group = self.args.max_len // seq_parallel_world_size
            local_seq_start = seq_parallel_world_rank * seq_len_per_group
            local_seq_end = (seq_parallel_world_rank +1) * seq_len_per_group
            input_ids = input_ids[:, local_seq_start:local_seq_end]
            labels = labels[:, local_seq_start:local_seq_end]
            freqs_cis = self.freqs_cis[local_seq_start:local_seq_end,:].to(input_ids.device)
        else:
            freqs_cis = self.freqs_cis.to(input_ids.device)
        freqs_cis.requires_grad_(True)
        return input_ids, labels, freqs_cis
    
    def embedding(self, input_ids):
        raise NotImplementedError()
    
    def model_forward(self, hidden_states, freqs_cis, attention_mask):
        raise NotImplementedError()
    
    def compute_loss(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
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
