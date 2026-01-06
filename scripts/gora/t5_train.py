import os
import sys
import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import DataLoader
from common.utils import set_random_seed
from common.lora_modules import *
from common.utils.params_manager import set_up_trainable_param

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.parser import get_args
from common.lora_modules.lora_set_up import *
from common.lora_modules.gora import *

class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_list, labels_list = [], []
        for instance in examples:
            input_ids = torch.LongTensor(instance["input_ids"]) if isinstance(instance["input_ids"], list) else instance["input_ids"]
            labels = torch.LongTensor(instance["labels"]) if isinstance(instance["labels"], list) else instance["labels"]
            input_ids_list.append(input_ids) 
            labels_list.append(labels)

        return {"input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list)}
        
class CustomTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        trainable_params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        if output_dir:
            output_dir, ckpt_name = output_dir.rsplit('/', maxsplit=1)
            ckpt_path = f"{output_dir}/{ckpt_name}.ckpt" 
        else:
            ckpt_path = "model_ckpt.ckpt"
        torch.save(trainable_params, ckpt_path)
        args_path = os.path.join(output_dir, "config.json")
        with open(args_path, "w") as f:
            save_dict = {k: v for k, v in args.__dict__.items() if k != 'device'}
            json.dump(save_dict, f)

    def _save_optimizer_and_scheduler(self, output_dir):
        return


def setup_optimizer(model, base_lr):
    weight_b_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'weight_b' in name:
            weight_b_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': weight_b_params, 'lr': base_lr * args.lora_plus_scaler},
        {'params': other_params, 'lr': base_lr},
    ])
    
    return optimizer

args = get_args()
set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = ""
tokenizer = T5Tokenizer.from_pretrained(model_name, attn_implementation='sdpa')

data_dir = ""

glue_tasks = {
    "MRPC": {
        "input_col": ["sentence1", "sentence2"], 
        "target_col": "label",
        "prefix": "mrpc sentence:",
        "label_map": {0: "not_equivalent", 1: "equivalent"}
    }
}

def preprocess_function(examples, task_name):
    task = glue_tasks[task_name]
    
    input_parts = []
    for i, col in enumerate(task["input_col"]):
        prefix = "" if i == 0 else f" {col}:"
        text = " ".join(examples[col]) if isinstance(examples[col], list) else str(examples[col])
        input_parts.append(prefix + " " + text)

    inputs = task["prefix"] + "".join(input_parts)
    if task['label_map']:
        targets = task["label_map"][examples[task["target_col"]]]
    else:
        targets = examples[task["target_col"]]

    model_inputs = tokenizer(inputs, max_length=args.max_src_len, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=args.max_len-args.max_src_len, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def train_and_evaluate(task_name):
    torch.cuda.empty_cache()
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    model.to(device)
    setup_lora(model, args)
    print(f"Training Task: {task_name}")
    
    data_files = {
        "train": os.path.join(data_dir, task_name, "train.tsv"),
        "validation": os.path.join(data_dir, task_name, "dev.tsv")
    }
    raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t", on_bad_lines="skip")
    
    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, task_name),
        batched=False
    )
    data_collator = DataCollator(tokenizer)
    train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=32, shuffle=True, collate_fn=data_collator)

    def forward_backward_func(model, batch):
        model

        loss = model(input_ids=batch['input_ids'],
                    labels=batch['labels']).loss
        return loss
    
    gora_reinit(model=model,
                dataloader=train_dataloader,
                args=args,
                iters=64,
                task_name=task_name,
                forward_backward_func=forward_backward_func)
    
    set_up_trainable_param(model, args)
    optimizer = setup_optimizer(model, 1e-4)
    
    training_args = TrainingArguments(
        output_dir=os.path.join("", args.experiment_name, task_name),
        eval_strategy="epoch",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=1,
        weight_decay=0,
        save_total_limit=1,
        save_steps=500,
        logging_steps=10,
        report_to="none",
        logging_dir=None,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        seed=args.seed,
    )
    

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None)
    )

    trainer.train()

train_and_evaluate("MRPC")
