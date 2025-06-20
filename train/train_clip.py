import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import json

from common.parser import get_args
from common.utils import set_random_seed
from common.lora_modules import *
from common.utils.params_manager import set_up_trainable_param
from common.utils.utils import print_rank_0

args = get_args()
set_random_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "/fs-computility/mabasic/yepeng/liminglei/cv/cv-ckpt/clip-vit-base-patch16"
DATASETS = ["stanford_cars", "dtd", "EuroSAT", "GTSRB", "RESISC45", "svhn"] # "stanford_cars" "dtd" "EuroSAT" "GTSRB" "RESISC45" "svhn"
TEST_SPLIT = 0.2

processor = CLIPProcessor.from_pretrained(MODEL_NAME, attn_implementation='sdpa')

def setup_optimizer(model, base_lr):
    """
    Set up optimizer with parameter groups based on the LoRA variant being used
    """
    lora_params = []
    other_params = []
    
    # Check for specific parameter types based on LoRA variant
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(lora_param in name for lora_param in ['weight_a', 'weight_b', 'lora_', 'lambda', 'shared_lora']):
                lora_params.append(param)
                print_rank_0(f"Found trainable LoRA param: {name} with shape {param.shape}", args.global_rank)
            else:
                other_params.append(param)
                print_rank_0(f"Found other trainable param: {name} with shape {param.shape}", args.global_rank)
    
    # Create parameter groups with appropriate learning rates
    param_groups = []
    
    if lora_params:
        param_groups.append({
            'params': lora_params, 
            'lr': base_lr * getattr(args, 'lora_plus_scaler', 1.0)
        })
        print_rank_0(f"LoRA parameter group created with {len(lora_params)} parameters", args.global_rank)
    
    if other_params:
        param_groups.append({
            'params': other_params, 
            'lr': base_lr
        })
        print_rank_0(f"Other parameter group created with {len(other_params)} parameters", args.global_rank)
    
    if not param_groups:
        raise ValueError("No trainable parameters found! This means the LoRA setup failed.")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    return optimizer

def prepare_datasets(name):
    folder = '/fs-computility/mabasic/yepeng/liminglei/cv/data/cv'
    label_name = "ClassId" if name == 'GTSRB' else "label"
    image_name = "Path" if name == 'GTSRB' else 'image'
    path = os.path.join(folder, name)
    if name == "svhn":
        dataset = load_dataset(path, "cropped_digits")
    elif name == "dtd":
        dataset = load_dataset(os.path.join(folder, 'dtd_train'))
    else:
        dataset = load_dataset(path)
    
    # Get label metadata
    label_features = dataset["train"].features[label_name]
    num_classes = label_features.num_classes
    class_names = label_features.names
    
    def preprocess(examples):
        images = [image.convert("RGB") for image in examples[image_name]]
        image_inputs = processor(
            images=images,
            return_tensors="pt",
            padding=True
        )
        
        return {
            "pixel_values": image_inputs.pixel_values,
            "labels": examples[label_name]
        }
    
    dataset = dataset.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=[image_name],
        num_proc=12
    )
    
    if "test" in dataset:
        train_data = dataset["train"]
        test_data = dataset["test"]
    elif name == 'dtd':
        train_data = dataset["train"]
        test_dataset = load_dataset(os.path.join(folder, 'dtd_test'))['test']
        test_data = test_dataset.map(
        preprocess,
        batched=True,
        batch_size=32,
        remove_columns=[image_name],
        num_proc=12
    )
    else:
        split = dataset["train"].train_test_split(test_size=TEST_SPLIT)
        train_data = split["train"]
        test_data = split["test"]
    
    train_data.set_format(type="torch", columns=["pixel_values", "labels"])
    test_data.set_format(type="torch", columns=["pixel_values", "labels"])
    
    print(f"Dataset {name} loaded. Train samples: {len(train_data)}, Test samples: {len(test_data)}")
    print(f"Label range: {min(train_data['labels'])}-{max(train_data['labels'])}, Num classes: {num_classes}")
    
    return train_data, test_data, class_names

def prepare_text_inputs(class_names):
    prompts = [f"a photo of a {name}" for name in class_names]
    text_inputs = processor.tokenizer(
        prompts, 
        padding='max_length', 
        return_tensors="pt",
        truncation=True,
        max_length=30
    )
    return text_inputs

class CLIPClassificationModel(nn.Module):
    def __init__(self, clip_model, class_names):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.device = clip_model.device
        
        self.text_inputs = prepare_text_inputs(class_names)
    
    def forward(self, pixel_values, labels=None):
        input_ids = self.text_inputs.input_ids.to(self.device)
        attention_mask = self.text_inputs.attention_mask.to(self.device)
        
        outputs = self.clip_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask
        )
        
        logits = outputs.logits_per_image
        
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return loss, logits

class CLIPClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        with torch.autocast(device_type='cuda', enabled=self.args.fp16):
            pixel_values = inputs["pixel_values"].to(model.device)
            labels = inputs["labels"].to(model.device)
            
            outputs = model(pixel_values, labels)
            loss = outputs[0]
        
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        pixel_values = inputs["pixel_values"].to(model.device)
        labels = inputs["labels"].to(model.device)
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', enabled=self.args.fp16):
                outputs = model(pixel_values, labels)
                loss = outputs[0]
                logits = outputs[1]
        
        if prediction_loss_only:
            return (loss.detach(), None, None)
        
        preds = torch.argmax(logits, dim=-1)
        return (loss.detach(), preds, labels)

    def save_model(self, output_dir=None, _internal_call=False):
        # Determine the base experiment directory using TrainingArguments output_dir and args.experiment_name
        # We ignore the 'output_dir' argument passed by the Trainer internally if it contains 'checkpoint-xxx'
        # as we want a fixed location.
        exp_dir = os.path.join(self.args.output_dir, args.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)

        # Save trainable parameters (LoRA weights and shared weights)
        trainable_params = {name: param for name, param in self.model.clip_model.named_parameters() if param.requires_grad}
        
        # Also save shared weights if they exist
        if hasattr(self.model.clip_model, 'shared_lora_A'):
            trainable_params['shared_lora_A'] = self.model.clip_model.shared_lora_A
        if hasattr(self.model.clip_model, 'shared_lora_B'):
            trainable_params['shared_lora_B'] = self.model.clip_model.shared_lora_B
        if hasattr(self.model.clip_model, 'vera_A'):
            trainable_params['vera_A'] = self.model.clip_model.vera_A.state_dict()
        if hasattr(self.model.clip_model, 'vera_B'):
            trainable_params['vera_B'] = self.model.clip_model.vera_B.state_dict()
        
        # Save with a fixed name instead of including step count
        ckpt_path = os.path.join(exp_dir, "lora_weights.ckpt")
        torch.save(trainable_params, ckpt_path)
        print(f"Saved LoRA weights to {ckpt_path}")

        # Save config in the same experiment directory
        args_path = os.path.join(exp_dir, "config.json")
        with open(args_path, "w") as f:
            # Filter out non-serializable items if necessary, like 'device'
            # save_dict = {k: v for k, v in args.__dict__.items() if k != 'device' and isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))}
            save_dict = {k: v for k, v in args.__dict__.items() if k != 'device' and isinstance(v, (str, int, float, bool, list, dict, tuple))}
            json.dump(save_dict, f, indent=2)
        print(f"Saved config to {args_path}")

    def _save_optimizer_and_scheduler(self, output_dir=None):
        # We ignore the 'output_dir' argument passed by the Trainer internally (which contains 'checkpoint-xxx')
        # Define the target directory fixed relative to the main output path and experiment name
        target_dir = os.path.join(self.args.output_dir, args.experiment_name, "checkpoint")
        os.makedirs(target_dir, exist_ok=True)

        # Save only random states and trainer state as requested
        rng_state_path = os.path.join(target_dir, "rng_state.pth")
        torch.save(torch.random.get_rng_state(), rng_state_path)
        print(f"Saved RNG state to {rng_state_path}")

        # Save trainer state
        if hasattr(self, 'state'):
            # Use vars() to get the state dictionary if possible
            try:
                state_dict_to_save = vars(self.state)
                trainer_state_path = os.path.join(target_dir, "trainer_state.json")
                with open(trainer_state_path, "w", encoding="utf-8") as f:
                    # Filter out potential non-serializable items just in case
                    serializable_state = {k: v for k, v in state_dict_to_save.items()
                                            if isinstance(v, (str, int, float, bool, list, dict, tuple, type(None)))}
                    json.dump(serializable_state, f, indent=2, sort_keys=True)
                print(f"Saved trainer state to {trainer_state_path}")
            except TypeError as e:
                 print(f"Warning: Could not serialize TrainerState using vars(): {e}. Skipping saving trainer_state.json.")

        # Removed saving of optimizer, scheduler, and cuda_rng_state

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    accuracy = accuracy_score(labels, preds)
    print(f"Test accuracy: {accuracy*100:.2f}%")
    return {"accuracy": accuracy}

def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

def print_trainable_parameters(model):
    """
    Print information about trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    shared_params = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            if 'shared_lora' in name or 'vera_A' in name or 'vera_B' in name:
                shared_params += param.numel()
            print_rank_0(f"Trainable parameter: {name}, shape: {param.shape}, numel: {param.numel()}", args.global_rank)
    
    print_rank_0(f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.2f}", args.global_rank)
    if shared_params > 0:
        print_rank_0(f"shared params: {shared_params:,} || shared%: {100 * shared_params / trainable_params:.2f}", args.global_rank)

def train_and_evaluate(dataset_name):
    clip_model = CLIPModel.from_pretrained(MODEL_NAME)
    clip_model.to(device)
    
    # Setup LoRA - modified to support multiple variants
    setup_lora(clip_model, args)
    
    train_dataset, test_dataset, class_names = prepare_datasets(dataset_name)
    
    model = CLIPClassificationModel(clip_model, class_names)
    model.to(device)

    # Create a temporary dataloader for initialization methods that need it
    temp_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    # Initialize any LoRA variants that need pre-training steps
    prepare_lora(model, temp_dataloader, args)
    
    # Set up trainable parameters
    set_up_trainable_param(clip_model, args)
    
    # Print trainable parameter information
    print_trainable_parameters(clip_model)
    
    # Setup optimizer with proper parameter groups
    optimizer = setup_optimizer(clip_model, args.lr)
    
    training_args = TrainingArguments(
        output_dir=args.output_path,
        per_device_train_batch_size=args.batch_size_per_gpu,
        per_device_eval_batch_size=args.eval_batch_size_per_gpu,
        num_train_epochs=max(1000 // len(temp_dataloader), 1),
        learning_rate=args.lr,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=1,
        save_total_limit=2,
        report_to="tensorboard" if args.tensorboard else "none",
        logging_dir=args.tb_log_dir,
        remove_unused_columns=False,
        seed=args.seed,
        warmup_ratio=0.03,
        lr_scheduler_type='cosine',
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True
    )

    trainer = CLIPClassificationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        optimizers=(optimizer, None)
    )

    print("Starting training...")
    trainer.train()

    eval_results = trainer.evaluate(test_dataset)
    print_rank_0(f"Test accuracy: {eval_results['eval_accuracy']*100:.2f}%")


if __name__ == "__main__":
    # Use the specified dataset if provided, otherwise use all
    datasets_to_train = [args.cv_dataset_name] if args.cv_dataset_name else DATASETS
    
    for dataset_name in datasets_to_train:
        print_rank_0(f"\n=== Processing dataset: {dataset_name} ===", args.global_rank)
        train_and_evaluate(dataset_name)