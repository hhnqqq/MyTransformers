#!/usr/bin/env python
# coding=utf-8

import os
import torch
import json
import logging
import numpy as np
import pandas as pd
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
    default_data_collator,
    DataCollatorWithPadding,
    EvalPrediction
)

# Import custom modules
from common.parser import get_args
from common.utils import set_random_seed
from common.lora_modules import *
from common.lora_modules.lora_set_up import setup_lora  # Explicit import to ensure we get the right functions
from common.utils.params_manager import set_up_trainable_param
from datetime import datetime
import wandb
import sys

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )

LORA_VARIANTS = {
    "use_dora": "dora",
    "use_hira": "hira",
    "use_mos_lora": "moslora",
    "use_me_lora": "melora",
    "use_lora_ga": "LoRAGA",
    "use_lora_one": "LoRAOne",
    "use_rslora": "rslora",
    "use_pissa": "pissa",
    "use_adalora": "adalora",
    "use_lora_moe": "moe",
    "use_randlora": "randlora",
    "use_goat": "goat",
    "use_rasa": "rasa",
    "use_dense_lora": "denselora",
    "use_eva": "EVA",
    "use_delora": "delora",
    "use_nzlora": "NZLoRA",
    "use_lora_plus": "lora+"
}
# relora将来单独跑

# Define task mappings
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# Define uppercase to lowercase task name mapping for consistency
task_name_mapping = {
    "SST-2": "sst2",
    "CoLA": "cola",
    "MNLI": "mnli",
    "MRPC": "mrpc",
    "QNLI": "qnli",
    "QQP": "qqp",
    "RTE": "rte",
    "STS-B": "stsb",
    "WNLI": "wnli"
}

# Define primary metrics for each GLUE task (for best model selection)
task_to_primary_metric = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "stsb": "pearson",
    "qqp": "f1",
    "mnli": "accuracy",
    "qnli": "accuracy", 
    "rte": "accuracy",
    "wnli": "accuracy"
}

# Define whether higher values are better for each metric
metric_greater_is_better = {
    "matthews_correlation": True,
    "accuracy": True,
    "f1": True,
    "pearson": True
}

need_label_map = {
    "MNLI": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "RTE": {"entailment": 0, "not_entailment": 1}
}

# Custom data collator for handling batches
class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids_list, attention_mask_list, labels_list = [], [], []
        for instance in examples:
            # logger.error(f"instance {type(instance)}\n{instance}")
            input_ids = torch.LongTensor(instance["input_ids"]) if isinstance(instance["input_ids"], list) else instance["input_ids"]
            attention_mask = torch.LongTensor(instance["attention_mask"]) if isinstance(instance["attention_mask"], list) else instance["attention_mask"]
            
            # Handle labels differently depending on type
            if isinstance(instance["labels"], (int, float)):
                labels = torch.tensor([instance["labels"]], dtype=torch.float if isinstance(instance["labels"], float) else torch.long)
            # elif isinstance(instance["labels"], (str)):
            else:
                labels = instance["labels"]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list)
        }

# Custom trainer for saving only trainable parameters and tracking best model
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.best_metric_value = None
        self.best_model_checkpoint = None
        self.best_metrics = None
        
    def save_model(self, output_dir=None, _internal_call=False):

        trainable_params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        if output_dir:
            output_dir, ckpt_name = output_dir.rsplit('/', maxsplit=1)
            ckpt_path = f"{output_dir}/{ckpt_name}.ckpt" 
            
            # Save configuration - only when output_dir exists
            if hasattr(self, 'args_dict'):
                args_path = os.path.join(output_dir, "config.json")
                with open(args_path, "w") as f:
                    save_dict = {k: v for k, v in self.args_dict.items() if k != 'device'}
                    json.dump(save_dict, f)
        else:
            ckpt_path = "model_ckpt.ckpt"
        
        dir_file = os.listdir(output_dir)
        for file in dir_file:
            if file.endswith(".ckpt"):
                os.remove(os.path.join(output_dir, file))
        
        torch.save(trainable_params, ckpt_path)

    def _save_optimizer_and_scheduler(self, output_dir):
        return
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """Override evaluate to track best model"""
        eval_result = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Check if this is the best model so far
        if hasattr(self, 'primary_metric'):
            metric_key = f"{metric_key_prefix}_{self.primary_metric}"
            if metric_key in eval_result:
                current_metric = eval_result[metric_key]
                is_better = False
                
                if self.best_metric_value is None:
                    is_better = True
                else:
                    if self.greater_is_better:
                        is_better = current_metric > self.best_metric_value
                    else:
                        is_better = current_metric < self.best_metric_value
                
                if is_better:
                    self.best_metric_value = current_metric
                    self.best_metrics = eval_result.copy()  # Store the best metrics
                    
                    # Save the best model
                    best_model_dir = os.path.join(self.args.output_dir, "best_model")
                    os.makedirs(best_model_dir, exist_ok=True)
                    self.save_model(os.path.join(best_model_dir, "model"))
                    
                    # Save best metrics
                    best_metrics_path = os.path.join(best_model_dir, "best_metrics.json")
                    with open(best_metrics_path, "w") as f:
                        json.dump(eval_result, f, indent=2)
                    
                    logger.info(f"New best model saved with {self.primary_metric}: {current_metric:.4f}")
        
        return eval_result
    
    def load_best_model(self):
        """Load the best model checkpoint"""
        best_model_dir = os.path.join(self.args.output_dir, "best_model")
        best_model_path = os.path.join(best_model_dir, "model.ckpt")
        
        if os.path.exists(best_model_path):
            logger.info(f"Loading best model from {best_model_path}")
            state_dict = torch.load(best_model_path, map_location=self.model.device)
            
            # Load only the trainable parameters
            model_state_dict = self.model.state_dict()
            for name, param in state_dict.items():
                if name in model_state_dict:
                    model_state_dict[name].copy_(param)
            
            logger.info(f"Best model loaded successfully with {self.primary_metric}: {self.best_metric_value:.4f}")
            return True
        else:
            logger.warning(f"Best model checkpoint not found at {best_model_path}")
            return False

def setup_optimizer(model, args):
    """Configure optimizer with different learning rates for different parameter groups"""
    base_lr = args.lr
    # Set default weight decay to 0.01 to match paper if not specified
    weight_decay = getattr(args, 'weight_decay', 0.01)
    
    if getattr(args, 'use_catvblora', False):
        shared_vector_bank_params = []
        block_vector_bank_params = []
        logits_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'shared_vector_bank' in name:
                    shared_vector_bank_params.append(param)
                elif 'block_vector_bank' in name:
                    block_vector_bank_params.append(param)
                elif 'logits_A' in name or 'logits_B' in name:
                    logits_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': shared_vector_bank_params, 'lr': args.learning_rate_vector_bank if hasattr(args, 'learning_rate_vector_bank') else base_lr},
            {'params': block_vector_bank_params, 'lr': args.learning_rate_vector_bank if hasattr(args, 'learning_rate_vector_bank') else base_lr},
            {'params': logits_params, 'lr': args.learning_rate_logits if hasattr(args, 'learning_rate_logits') else base_lr},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=weight_decay)
    elif getattr(args, 'use_vblora', False) or getattr(args, 'use_g_vblora', False):
        vector_bank_params = []
        logits_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'vector_bank' in name:
                    vector_bank_params.append(param)
                elif 'logits_A' in name or 'logits_B' in name:
                    logits_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': vector_bank_params, 'lr': args.learning_rate_vector_bank if hasattr(args, 'learning_rate_vector_bank') else base_lr},
            {'params': logits_params, 'lr': args.learning_rate_logits if hasattr(args, 'learning_rate_logits') else base_lr},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=weight_decay)
    elif getattr(args, 'use_entropy_interval_vblora', False) or getattr(args, 'use_entropy_graph_interval_vblora', False):
        # Fixed parameter grouping for entropy interval VBLoRA variants
        vector_bank_params = []
        logits_params = []
        other_params = []
        
        # Debug: Print all parameter names to see what we have
        logger.info("Entropy VBLoRA parameter analysis:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"  {name}: {param.shape}")
                if 'partition_vector_bank' in name or 'center_vector_bank' in name or 'vector_bank' in name:
                    vector_bank_params.append(param)
                    logger.info(f"    -> Added to vector_bank_params")
                elif 'logits_A' in name or 'logits_B' in name:
                    logits_params.append(param)
                    logger.info(f"    -> Added to logits_params")
                else:
                    other_params.append(param)
                    logger.info(f"    -> Added to other_params")
        
        logger.info(f"Parameter groups: vector_bank={len(vector_bank_params)}, logits={len(logits_params)}, other={len(other_params)}")
        
        # Create parameter groups with different learning rates
        param_groups = []
        if vector_bank_params:
            param_groups.append({
                'params': vector_bank_params, 
                'lr': getattr(args, 'learning_rate_vector_bank', base_lr)
            })
            logger.info(f"Vector bank LR: {getattr(args, 'learning_rate_vector_bank', base_lr)}")
        
        if logits_params:
            param_groups.append({
                'params': logits_params, 
                'lr': getattr(args, 'learning_rate_logits', base_lr)
            })
            logger.info(f"Logits LR: {getattr(args, 'learning_rate_logits', base_lr)}")
        
        if other_params:
            param_groups.append({
                'params': other_params, 
                'lr': base_lr
            })
            logger.info(f"Other params LR: {base_lr}")
        
        optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif getattr(args, 'use_lora_plus', False):
        weight_b_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if 'weight_b' in name:
                    weight_b_params.append(param)
                else:
                    other_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': weight_b_params, 'lr': base_lr * args.lora_plus_scaler},
            {'params': other_params, 'lr': base_lr},
        ], weight_decay=weight_decay)
    else:
        # Changed from Adam to AdamW
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], 
            lr=base_lr,
            weight_decay=weight_decay
        )
    
    return optimizer

def main():
    # Get arguments
    args = get_args()

    set_random_seed(args.seed)

    # Set up logger
    log_base_dir = args.output_path.rsplit('/', maxsplit=2)[0]
    log_dir_path = os.path.join(log_base_dir, 'results')
    os.makedirs(log_dir_path, exist_ok=True)

    timestamp_for_log = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{args.experiment_name}_{timestamp_for_log}.log" 
    log_file_path = os.path.join(log_dir_path, log_file_name)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING) # 控制台记录 WARNING 及以上级别的日志

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S")
    
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalize task name if provided
    if args.task_name:
        task_name_list = args.task_name
        for task_name in task_name_list:
            # Convert to lowercase if it's one of our mappings
            if task_name in task_name_mapping:
                # Nothing to do
                continue
            else:
                # Try direct lowercase
                glue_task_name = task_name.lower()
                # Verify it's a valid task
                if glue_task_name not in task_to_keys:
                    raise ValueError(f"Task {task_name} not recognized. Valid tasks are: {list(task_name_mapping.keys())}")
    else:
        logger.info("No task specified, will run all GLUE tasks")
    
    # Load tokenizer
    model_name = args.model_name_or_path if args.model_name_or_path else "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Log setup information
    logger.info(f"Using model: {model_name}")
    logger.info(f"Using device: {device}")
    logger.info(f"Arguments loaded: task_name={args.task_name}, seed={args.seed}, learning_rate={args.lr}")
    
    if hasattr(args, 'use_vblora') and args.use_vblora:
        logger.info(f"VBLoRA config: num_vectors={args.num_vectors}, vector_length={args.vector_length}, topk={args.topk}")
    elif hasattr(args, 'use_g_vblora') and args.use_g_vblora:
        logger.info(f"G_VBLoRA config: num_vectors={args.num_vectors}, vector_length={args.vector_length}, topk={args.topk}")
    elif hasattr(args, 'use_entropy_interval_vblora') and args.use_entropy_interval_vblora:
        logger.info(f"EntropyIntervalVBLoRA config: num_vectors_per_bank={args.num_vectors_per_bank}, vector_length={args.vector_length}, topk={args.topk}")
    
    # Define tasks to train on
    if args.task_name:
        tasks_to_train = args.task_name
    else:
        # Default to all GLUE tasks
        tasks_to_train = list(task_name_mapping.keys())
    
    # Train on each task
    for task in tasks_to_train:
        train_and_evaluate(task, args, device, tokenizer, model_name)
    logger.info("All tasks completed.")
    logger.info("Exiting program.")
    sys.exit(0)

def get_num_labels(task_name):
    """Get the number of labels for a task"""
    lowercase_task = task_name.lower() if task_name not in task_name_mapping else task_name_mapping[task_name]
    
    if lowercase_task == "mnli":
        return 3
    elif lowercase_task == "stsb":
        # STS-B is a regression task
        return 1
    else:
        return 2

def is_regression_task(task_name):
    """Check if a task is a regression task"""
    lowercase_task = task_name.lower() if task_name not in task_name_mapping else task_name_mapping[task_name]
    return lowercase_task == "stsb"

# Add this function after the existing imports and before the main() function
def update_results_csv(task_name, metric_value, variant_name, csv_path="/home/bingxing2/ailab/hehaonan/workspace/MyTransformers/Roberta/glue_results.csv"):
    """Update the results CSV file with new evaluation results"""
    
    # Define the structure
    variant_order = [
        'LoRA', 'LoRAGA', 'NZLoRA', 'LoRAOne', 'EVA', 'lora+', 'hira', 'rasa', 
        'denselora', 'randlora', 'melora', 'relora', 'dora', 'moe', 'delora', 
        'moslora', 'rslora', 'goat', 'adalora', 'pissa'
    ]
    
    # Task name mapping to CSV columns
    task_to_csv_column = {
        'SST-2': 'SST-2',
        'CoLA': 'CoLA(MC)', 
        'MNLI': 'MNLI',
        'MRPC': 'MRPC',
        'QNLI': 'QNLI',
        'QQP': 'QQP(F1)',
        'RTE': 'RTE', 
        'STS-B': 'STS-B(person)',
        'WNLI': 'WNLI'
    }
    
    # Get the CSV column name
    csv_column = task_to_csv_column.get(task_name, task_name)
    
    # Try to load existing CSV or create new one
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        # Create new DataFrame with the specified structure
        columns = ['SST-2', 'CoLA(MC)', 'MNLI', 'MRPC', 'QNLI', 'QQP(F1)', 'RTE', 'STS-B(person)', 'WNLI', 'AVG']
        df = pd.DataFrame(index=variant_order, columns=columns)
    
    # Update the specific cell
    df.loc[variant_name, csv_column] = metric_value
    
    # Calculate average if all tasks are completed for this variant
    task_columns = ['SST-2', 'CoLA(MC)', 'MNLI', 'MRPC', 'QNLI', 'QQP(F1)', 'RTE', 'STS-B(person)', 'WNLI']
    if df.loc[variant_name, task_columns].notna().all():
        df.loc[variant_name, 'AVG'] = df.loc[variant_name, task_columns].astype(float).mean()
    
    # Save the updated CSV
    df.to_csv(csv_path)
    logger.info(f"Updated results CSV: {variant_name} - {csv_column} = {metric_value:.4f}")

def train_and_evaluate(task_name, args, device, tokenizer, model_name):
    """Train and evaluate a model on a specific GLUE task"""
    torch.cuda.empty_cache()
    
    # Map task name to glue format
    glue_task_name = task_name.lower() if task_name not in task_name_mapping else task_name_mapping[task_name]
    logger.info(f"Training Task: {task_name} (GLUE task: {glue_task_name})")
    
    # Check if it's a regression task
    is_regression = is_regression_task(task_name)
    
    # Load the dataset from the Hugging Face datasets hub
    logger.info(f"Loading dataset for task: {task_name}")
    # try:
    #     raw_datasets = load_dataset("glue", glue_task_name)
    #     logger.info(f"Successfully loaded {glue_task_name} from Hugging Face datasets hub")
    # except Exception as e:
    #     logger.error(f"Error loading dataset from hub: {e}")
    #     logger.info("Attempting to load from local files...")
    try:
        data_dir = "/home/bingxing2/ailab/hehaonan/workspace/MyTransformers/Roberta/GLUE"
        data_files = {
            "train": os.path.join(data_dir, task_name, "train.tsv"),
            # "validation": os.path.join(data_dir, task_name, "mnli_validation_matched_empty_idx_header.tsv")
            "validation": os.path.join(data_dir, task_name, "dev.tsv")
        }
        
        if os.path.exists(data_files["train"]) and os.path.exists(data_files["validation"]):
            logger.info(f"Loading from local files: {data_files}")
            raw_datasets = load_dataset("csv", data_files=data_files, delimiter="\t", on_bad_lines="skip")
        else:
            raise FileNotFoundError(f"Could not find files at {data_files}")
    except Exception as load_error:
        logger.error(f"Failed to load dataset: {load_error}")
        return
    
    # Get the input column names based on the task
    if glue_task_name in task_to_keys:
        sentence1_key, sentence2_key = task_to_keys[glue_task_name]
    else:
        # This shouldn't happen since we validate the task name earlier
        logger.error(f"Task {glue_task_name} not found in task_to_keys")
        return
    
    # Initialize the model
    num_labels = get_num_labels(task_name)
    model = RobertaForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="regression" if is_regression else "single_label_classification"
    )
    model.to(device)
    
    # Setup LoRA if enabled
    logger.info("Setting up LoRA...")
    setup_lora(model, args)
    
    # Define preprocessing function
    def preprocess_function(examples):
        # Tokenize the texts
        texts = (
            (examples[sentence1_key],) if sentence2_key is None 
            else (examples[sentence1_key], examples[sentence2_key])
        )
        
        # Tokenize all texts - use 128 for most GLUE tasks per paper
        # Use the provided max_src_len if available, otherwise use 128 as default for GLUE
        seq_length = getattr(args, 'max_src_len', 128)
        # For GLUE tasks, override to 128 as per the paper unless specifically set otherwise
        if not hasattr(args, 'override_seq_length') and task_name in task_name_mapping:
            seq_length = min(seq_length, 128)
            
        result = tokenizer(
            *texts,
            padding="max_length",
            max_length=seq_length,
            truncation=True
        )
        
        # Map labels
        if "label" in examples:
            if is_regression:
                # For regression tasks
                result["labels"] = examples["label"]
            else:
                # For classification tasks
                # label = [example for example in examples["label"]]
                # logger.info(f"exaples: {label}")
                if task_name in need_label_map.keys():
                    map_dict = need_label_map[task_name]
                    # logger.info(f"map_dict: {map_dict}")
                    result["labels"] = [int(map_dict[example]) if example in map_dict else int(example) for example in examples['label']]
                    # logger.info(f"map_dict: {list(zip(result['labels'], examples['label']))}")
                    # result["labels"] = []
                    # for i, example in enumerate(examples['label']):
                    #     if example is not None:
                    #         result["labels"].append(int(map_dict[example.lower()]) if example in map_dict else int(example))
                    #     else:
                            
                        # try:
                        #     if example.lower() in map_dict:
                        #         result["labels"].append(int(map_dict[example.lower()]))
                        #     else:
                        #         result["labels"].append(int(example))
                        # except Exception as e:
                        #     logger.error(f"Error find when processing example: {examples['sentence1'][i]}")
                        #     logger.error(f"Error find when processing example: {examples['sentence2'][i]}")
                        #     print(e)
                else:
                    result["labels"] = examples["label"]
        elif "is_duplicate" in examples:
            result["labels"] = examples["is_duplicate"]
        return result
    
    # Preprocess the datasets
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    
    # Prepare train dataset
    train_dataset = processed_datasets["train"]
    
    # Prepare validation dataset
    eval_dataset = processed_datasets["validation"]

    eval_datasets = {"default": eval_dataset}
    # Setup data collator
    data_collator = DataCollator(tokenizer)
    
    # Create dataloader for initialization (required for LoRA variants like GoRA or G_VBLoRA)
    batch_size = getattr(args, 'batch_size_per_gpu', 32)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=data_collator
    )
    
    # Initialize LoRA variants that need pre-training steps
    logger.info("Preparing LoRA initialization...")
    prepare_lora(model, train_dataloader, tokenizer, args)
    
    # Setup trainable parameters
    logger.info("Setting up trainable parameters...")
    set_up_trainable_param(model, args)
    
    # Log number of trainable parameters
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")
    
    # Setup optimizer AFTER LoRA initialization
    logger.info("Setting up optimizer...")
    optimizer = setup_optimizer(model, args)
    
    # Configure training arguments
    num_epochs = args.epochs if hasattr(args, 'epochs') else 3
    warmup = args.warmup if hasattr(args, 'warmup') else 0.1
    logging_steps = args.show_loss_step if hasattr(args, 'show_loss_step') else 10
    
    output_dir = os.path.join(args.output_path, task_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine the learning rate scheduler type from args - default to linear as in the paper
    lr_scheduler_type = getattr(args, 'lr_decay_style', 'linear')
    
    # Handle precision settings
    fp16 = False
    bf16 = False
    if hasattr(args, 'bf16') and args.bf16:
        bf16 = True
    # Only enable fp16 if explicitly specified
    elif hasattr(args, 'fp16') and args.fp16 and torch.cuda.is_available():
        fp16 = True

    if args.wandb:
        current_time = datetime.now().strftime('%y-%m-%d_%H-%M')
        os.environ['WANDB_CACHE_DIR'] = args.wandb_cache_dir
        os.environ['WANDB_DIR'] = args.wandb_dir
        os.environ["WANDB_PROJECT"]=args.wandb_project
        if args.wandb_api_key:
            os.environ['WANDB_API_KEY'] = args.wandb_api_key
        wandb.init(project=args.wandb_project,
                        entity=args.wandb_team,
                        name=args.experiment_name + current_time,
                        config=args)

    # Get the primary metric for this task
    primary_metric = task_to_primary_metric.get(glue_task_name, "accuracy")
    greater_is_better = metric_greater_is_better.get(primary_metric, True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        eval_strategy ="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=getattr(args, 'eval_batch_size_per_gpu', batch_size),
        learning_rate=args.lr,
        weight_decay=getattr(args, 'weight_decay', 0.01),  # Default to 0.01 if not specified
        num_train_epochs=num_epochs,
        warmup_ratio=warmup,
        lr_scheduler_type=lr_scheduler_type,  # Use the value from args
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_total_limit=1,  # Keep only 1 regular checkpoint
        seed=args.seed,
        fp16=False, # To use fp32
        bf16=False,
        # Disable automatic best model loading since we'll do it manually
        load_best_model_at_end=False,
        metric_for_best_model=primary_metric,
        greater_is_better=greater_is_better,
        report_to = 'wandb',
        run_name=args.experiment_name+task_name,
    )
    
    # Get metrics function
    # metric = evaluate.load(primary_metric)
    metric = evaluate.load("glue", glue_task_name)
    
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
            
        result = metric.compute(predictions=preds, references=p.label_ids)
        
        # Add combined score for tasks with multiple metrics
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        
        return result
    
    # Create the trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets["default"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)
    )
    
    # Store args for saving with the model
    trainer.args_dict = vars(args)
    # Store metric info for best model tracking
    trainer.primary_metric = primary_metric
    trainer.greater_is_better = greater_is_better
    
    # Train the model
    logger.info("Starting training")
    train_result = trainer.train()
    
    # Save final model (this is the last checkpoint)
    # trainer.save_model()
    
    # Log and save training metrics
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Load the best model for final evaluation
    # best_model_loaded = trainer.load_best_model()
    best_model_loaded = False
    
    if best_model_loaded and trainer.best_metrics:
        # Evaluate on all validation sets using the BEST model
        logger.info("=" * 80)
        logger.info(f"FINAL EVALUATION USING BEST MODEL (Best {primary_metric}: {trainer.best_metric_value:.4f})")
        logger.info("=" * 80)
        
        for eval_name, eval_dataset in eval_datasets.items():
            logger.info(f"*** Evaluating BEST model on {task_name} {eval_name} ***")
            eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=f"final_eval_{eval_name}")
            
            # Log the metrics with clear indication that these are from the best model
            logger.info(f"*** BEST MODEL {eval_name.upper()} RESULTS ***")
            for key, value in eval_metrics.items():
                logger.info(f"key: {key}")
                if key.startswith(f"final_eval_{eval_name}_"):
                    metric_name = key.replace(f"final_eval_{eval_name}_", "")
                    logger.info(f"  BEST {metric_name} = {value:.4f}")
            
            trainer.log_metrics(f"best_model_eval_{eval_name}", eval_metrics)
            trainer.save_metrics(f"best_model_eval_{eval_name}", eval_metrics)
        
        # Print summary of best results
        logger.info("=" * 80)
        logger.info("BEST MODEL SUMMARY:")
        logger.info(f"  Best {primary_metric} achieved: {trainer.best_metric_value:.4f}")
        logger.info(f"  Best model saved in: {os.path.join(output_dir, 'best_model')}")
        logger.info("=" * 80)
    else:
        # logger.warning("Could not load best model, using final checkpoint for evaluation and test")
        # Fallback to regular evaluation with final checkpoint
        for eval_name, eval_dataset in eval_datasets.items():
            # logger.info(f"*** Evaluating and Testing final model on {task_name} {eval_name} ***")
            # eval_metrics = trainer.evaluate(eval_dataset=eval_dataset, metric_key_prefix=f"final_eval_{eval_name}")
            # for key, value in eval_metrics.items():
            #     if key.startswith(f"final_eval_{eval_name}_"):
            #         metric_name = key.replace(f"final_eval_{eval_name}_", "")
            #         logger.info(f"  BEST {metric_name} = {value:.4f}")
            # trainer.log_metrics(f"eval_{eval_name}", eval_metrics)
            # trainer.save_metrics(f"eval_{eval_name}", eval_metrics)
            logger.info(f"*** Testing final model on {task_name} {eval_name} ***")
            test_metrics = trainer.predict(test_dataset=eval_dataset, metric_key_prefix=f"final_test_{eval_name}")
            logger.info(f"*** MODEL {eval_name.upper()} TEST RESULTS ***")

            # Extract the primary metric value for CSV logging
            primary_metric_key = f"final_test_{eval_name}_{primary_metric}"
            primary_metric_value = None

            for key, value in test_metrics.metrics.items():
                if key.startswith(f"final_test_{eval_name}_"):
                    metric_name = key.replace(f"final_test_{eval_name}_", "")
                    logger.info(f"  BEST {metric_name} = {value:.4f}")

                    # Capture the primary metric value
                    if key == primary_metric_key:
                        primary_metric_value = value
            # Update CSV with results
            if primary_metric_value is not None:
                # Extract variant name from args (you may need to adjust this based on your argument structure)
                variant_name = "LoRA"
                for k, v in LORA_VARIANTS.items():
                    if getattr(args, k, False):
                        variant_name = v
                        break
                
                # variant_name = getattr(args, 'variant_name', 'LoRA')  # Default to 'LoRA' if not specified
                results_csv_path = f'/home/bingxing2/ailab/hehaonan/workspace/MyTransformers/Roberta/glue_results{args.lr}.csv'
                update_results_csv(task_name, primary_metric_value, variant_name, csv_path=results_csv_path)
    
    logger.info(f"Training completed for {task_name}")

if __name__ == "__main__":
    main()