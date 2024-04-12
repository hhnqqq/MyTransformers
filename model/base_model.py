import torch
import torch.nn as nn
import common.utils.parallel_states as parallel_states

class BaseModel(nn.Module):
    def __init__(self, args, pad_id):
        super().__init__()
        self.args = args
        self.pad_id = pad_id
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, input_ids, labels):
        input_ids, labels, freqs_cis = self.cut_sequence(input_ids, labels)
        hidden_states, attention_mask = self.embedding(input_ids)
        logits = self.model_forward(hidden_states, freqs_cis, attention_mask)
        loss = self.compute_loss(logits, labels)
        return loss
    
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