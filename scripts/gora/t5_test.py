import os
import sys
import torch
import numpy as np
import argparse
import json
from common.lora_modules import *
from common.utils import load_ckpt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from datasets import load_dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.lora_modules.lora_set_up import *
from common.lora_modules.gora import *

def parse_args():
    parser = argparse.ArgumentParser(description="Test T5 model on GLUE tasks")
    parser.add_argument("--checkpoint_path", type=str, required=True, 
                        help="Base path to the checkpoint directory")
    parser.add_argument("--local_rank", type=int)
    return parser.parse_args()

cli_args = parse_args()

glue_tasks = {
    "MRPC": {
        "input_col": ["sentence1", "sentence2"], 
        "target_col": "label",
        "prefix": "mrpc sentence1:",
        "label_map": {0: "not_equivalent", 1: "equivalent"},
        "checkpoint": os.path.join(cli_args.checkpoint_path, "MRPC")
    }
}

tokenizer = T5Tokenizer.from_pretrained("", local_files_only=True)

def find_max_step_checkpoint(checkpoints):
    def extract_step(checkpoint):
        return int(checkpoint.split('-')[1].split('.')[0])
    max_checkpoint = max(checkpoints, key=extract_step)
    return max_checkpoint

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

    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=32, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

def compute_accuracy(predictions, labels):
    if isinstance(predictions[0], (int, np.integer)):
        # 数字标签
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    else:
        # 文本标签
        correct = sum(1 for pred, label in zip(predictions, labels) if str(pred).strip().lower() == str(label).strip().lower())
    
    accuracy = correct / len(predictions) if len(predictions) > 0 else 0.0
    return accuracy

def evaluate_on_test_set(task_name, batch_size=32):
    torch.cuda.empty_cache()
    task = glue_tasks[task_name]
    checkpoint_path = task["checkpoint"]
    ckpts = [i for i in os.listdir(checkpoint_path) if i.endswith('.ckpt')]
    ckpt = find_max_step_checkpoint(ckpts)
    model_ckpt_path = os.path.join(checkpoint_path, ckpt)
    args = argparse.Namespace(**json.load(open(os.path.join(checkpoint_path, 'config.json'), "r")))
    args.device = "cuda"
    model = T5ForConditionalGeneration.from_pretrained("")

    setup_lora(model, args)
    if args.use_gora:
        rank_config_file = os.path.join(checkpoint_path, 'rank.json')
        rank_config = json.load(open(rank_config_file, 'r'))
        for name, module in model.named_modules():
            if isinstance(module, LinearWithGoRA):
                rank = rank_config[name]
                module.init_method = 'vanilla'
                module.dynamic_init(args.lora_rank, rank)

    load_ckpt(model, partial_ckpt_path=model_ckpt_path, ignore_incompatible=True)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    data_files = {"test": os.path.join("", task_name, "dev.tsv")}
    raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t", on_bad_lines="skip")

    tokenized_datasets = raw_datasets.map(
        lambda x: preprocess_function(x, task_name),
        batched=False
    )
    
    predictions = []
    labels = []

    def collate_fn(batch):
        """Collate function for DataLoader to handle tokenized datasets."""
        input_ids = torch.tensor([example["input_ids"] for example in batch])
        labels = torch.tensor([example["labels"] for example in batch])
        return {"input_ids": input_ids, "labels": labels}

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets["test"],
        batch_size=batch_size,
        collate_fn=collate_fn
    )

    for batch in tqdm(dataloader, desc=f"Evaluating {task_name}"):
        input_ids = batch["input_ids"].to(model.device)
        label_ids = batch["labels"].to(model.device)

        with torch.no_grad():
            output_ids = model.generate(input_ids, max_length=32)

        pred_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        label_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in label_ids]
        pred_labels = []
        
        if task["label_map"] is not None:
            for text in pred_texts:
                try:
                    pred_label = next(k for k, v in task["label_map"].items() if v == text)
                    pred_labels.append(pred_label)
                except StopIteration:
                    pred_labels.append(-1)

            true_labels = [next(k for k, v in task["label_map"].items() if v == text) for text in label_texts]
        else:
            pred_labels = pred_texts
            true_labels = label_texts

        predictions.extend(pred_labels)
        labels.extend(true_labels)
    
    accuracy = compute_accuracy(predictions, labels)
    print(f"Test Accuracy for {task_name}: {accuracy:.4f}")
    
    total = len(predictions)
    correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    print(f"  Correct: {correct}/{total}")
    
    return accuracy

def main():
    accuracies = {}
    print("start")
    for task_name in ["MRPC"]:
        accuracy = evaluate_on_test_set(task_name)
        accuracies[task_name] = accuracy
    
    print("\n" + "="*50)
    print("Summary of Test Accuracies:")
    print("="*50)
    for task_name, acc in accuracies.items():
        print(f"{task_name:10s}: {acc:.4f}")

if __name__ == "__main__":
    main()
