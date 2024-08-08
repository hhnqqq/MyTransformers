import os
import gc
import io
import sys
import time
import json
import random
import logging
import contextlib
import configparser
from typing import Optional, Dict
from datetime import datetime
from traceback import format_exc
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score

import pytz
import torch
import deepspeed
import immutabledict
import numpy as np
import torch.distributed as dist
from torch.nn import Module
from dataclasses import make_dataclass
import torch.nn.utils.rnn as rnn_utils

import common.utils.parallel_states as parallel_states

seed_set = False

def is_seed_set():
    return seed_set

def dict_to_dataclass(name, data_dict):
    fields = [(key, type(value)) for key, value in data_dict.items()]
    DataClass = make_dataclass(name, fields)
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
            nested_name = f"{name}_{key.capitalize()}"
            nested_dataclass = dict_to_dataclass(nested_name, value)
            setattr(DataClass, key, nested_dataclass)
    
    return DataClass(**data_dict)

def cal_metric(y_true, y_pred):
    return {"accuracy":accuracy_score(y_true, y_pred),
            "f1":f1_score(y_true, y_pred, average="macro", zero_division=0),
            "mcc":matthews_corrcoef(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),}


@contextlib.contextmanager
def ignore_module_print():
    """
    A context manager that redirects stdout to devnull.
    """
    save_stdout = sys.stdout
    sys.stdout = io.StringIO()
    yield
    sys.stdout = save_stdout

class Timer(object):
    def __init__(self, start=None, n_round=2, iterations: Optional[int] = None):
        """
        A timer environment for loop programs.

        Args:
            start (time): Start time for the timer. If None, the current time is used.
            n_round (int): Number of decimal places to keep for time values.
            iterations (Optional[int]): The total number of iterations the loop will perform.
        """
        self.round = n_round  # Number of decimal places for time values
        self.start = round(start if start is not None else time.time(), self.round)  # Start time of the timer
        self.loop_start = None  # Start time of the current loop iteration
        self.iterations_left = iterations  # Number of iterations left

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit method.
        Returns:
            bool: True if no exception occurred, False otherwise.
        """
        self.stop = round(time.time(), self.round)  # Stop time of the timer
        self.time_cost = self.format_time(round(self.stop - self.start, self.round))  # Total time cost
        return exc_type is None  # Return True if no exception occurred

    def average_time(self, entry):
        """
        Records the start or end time of a loop iteration.

        Args:
            entry (str): Either 'start' to record the start time or 'end' to record the end time.

        Raises:
            ValueError: If entry is not 'start' or 'end'.
            AssertionError: If 'end' is called before 'start'.
        """
        current_time = round(time.time(), self.round)  # Current time
        if entry == 'start':
            if self.loop_start is None:
                self.loop_start = current_time  # Record the start time of the loop iteration
        elif entry == 'end':
            assert self.loop_start is not None, 'Please ensure average_time("start") is used before average_time("end")'
            if self.iterations_left is not None:
                self.iterations_left -= 1  # Decrement the number of iterations left
            loop_end = current_time
            self.loop_time = round(loop_end - self.loop_start, self.round)  # Calculate the time taken for the loop iteration
            self.loop_start = None  # Reset the loop start time
        else:
            raise ValueError("Invalid entry value. Expected 'start' or 'end'.")

    def calculate_remaining_time(self):
        """
        Calculates the remaining time based on the average time per iteration and the number of iterations left.

        Returns:
            str: Formatted remaining time.
        """
        total_time_seconds = self.iterations_left * self.loop_time  # Total remaining time in seconds
        return self.format_time(total_time_seconds)

    def format_time(self, input_time):
        if input_time < 60:
            return f"{round(input_time, self.round)}s"  # Less than a minute
        elif input_time < 3600:
            minutes = input_time // 60
            seconds = input_time % 60
            return f"{minutes}min {round(seconds, self.round)}s"  # Less than an hour
        elif input_time < 86400:
            hours = input_time // 3600
            minutes = (input_time % 3600) // 60
            seconds = (input_time % 3600) % 60
            return f"{hours}h {minutes}min {round(seconds, self.round)}s"  # Less than a day
        else:
            days = input_time // 86400
            hours = (input_time % 86400) // 3600
            minutes = ((input_time % 86400) % 3600) // 60
            seconds = ((input_time % 86400) % 3600) % 60
            return f"{days}d {hours}h {minutes}min {round(seconds, self.round)}s"  # More than a day

def read_config(file_path, encoding='utf-8'):
    """
    Read config file.
    """
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding=encoding) as f:
            config = json.load(f)
    elif file_path.endswith('.ini'):
        config = configparser.ConfigParser()
        config.read(file_path)
    else:
        if '.' in file_path:
            format = file_path.split('.')[-1]
        else:
            format = 'Unkown'
        raise ValueError(f"Can not read unsupported file format: {format}")
    return config

def ensure_directory_exists(directory, global_rank=0):
    if not os.path.exists(directory) and global_rank == 0: # Only create dir when global rank is 0.
        os.makedirs(directory)
        print(f'---> Directory:{directory} is not existed. created a new floder')

def count_trainable_parameters(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([torch.numel(p) for p in trainable_params])
    return num_trainable_params

def to_device(batch, device):
    """
    Move every pytorch tensor in the batch data to device for training.
    """
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class DataCollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_id

    def __call__(self, examples):
        input_ids_list, labels_list, cal_metric_pos_list, dna_ids_list, before_dna_list = [], [], [], [], []
        for instance in examples:
            input_ids = instance["input_ids"]
            labels = instance["labels"]
            cal_metric_pos = instance["cal_metric_pos"]
            dna_ids = instance.get("dna_ids", None)
            before_dna= instance.get("before_dna", None)
            input_ids_list.append(input_ids) 
            labels_list.append(labels)
            cal_metric_pos_list.append(cal_metric_pos)
            dna_ids_list.append(dna_ids)
            before_dna_list.append(before_dna)

        if None in cal_metric_pos_list:
            cal_metric_pos_list = None
        if None in dna_ids_list or None in before_dna_list:
            dna_ids_list = None
            before_dna_list = None

        return {"input_ids": torch.stack(input_ids_list),
                "dna_ids": rnn_utils.pad_sequence(dna_ids_list, batch_first=True, padding_value=4) if dna_ids_list is not None else None,
                "labels": torch.stack(labels_list),
                "cal_metric_pos_tensor": torch.tensor(cal_metric_pos_list) if cal_metric_pos_list is not None else None,
                "before_dna": torch.tensor(before_dna_list) if before_dna_list is not None else None}

class PipeLine_Datacollator():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_id

    def __call__(self, examples):
        input_ids_list, labels_list = [], []
        for instance in examples:
            input_ids_list.append(instance["input_ids"])
            labels_list.append(instance["labels"])
        return ((torch.stack(input_ids_list), torch.stack(labels_list)), torch.stack(labels_list))

def set_random_seed(seed):
    global seed_set
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        seed_set = True

def init_dist(args):
    if args.device == 'cuda':
        if args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            deepspeed.init_distributed(dist_backend="nccl")
            args.world_size = dist.get_world_size()
            args.global_rank = dist.get_rank()
    else:
        device = 'cpu'
    if args.num_sp_stages:
        assert args.atten_type == 'ulysses_atten', 'when using sequence parallism, the attention type must be `ulysses_atten`'
        parallel_states.initialize_model_parallel(sequence_model_parallel_size=args.num_sp_stages)
    elif args.num_pp_stages:
        parallel_states.initialize_model_parallel(pipeline_model_parallel_size=args.num_pp_stages)
    else:
        parallel_states.initialize_model_parallel()
    args.device = device
    return args
    
    

# ------------------logging----------------------------
def configure_logging(log_path, rank: Optional[int] = 0):
    """
    Configure logging functionality.

    Args:
        log_path (str): Path where the log files will be stored.
        rank (optional[int]): Rank used for creating directories, default is 0.

    Returns:
        logger (logging.Logger): Configured logger object.
    """
    # Get environment variables.
    sh_level = os.environ.get("PRINTLEVEL", logging.DEBUG)
    fh_level = os.environ.get("LOGLEVEL", logging.INFO)
    fh_disable = os.environ.get("NO_LOG_FILE", "false") == 'true' # Convert string variable to boolean

    logger = logging.getLogger("DNALLaMa")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s') # Define the log format
    
    # Create a console log handler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(sh_level)
    logger.addHandler(sh)

    timezone = pytz.timezone('Asia/Shanghai')
    # Get current date and time strings, formatted as 'yy-mm-dd' and 'HH-MM.log'
    date_string, hour_string = datetime.now().strftime('%y-%m-%d'), datetime.now().strftime('%H-%M') + '.log'
    log_path = os.path.join(log_path, date_string)

    ensure_directory_exists(log_path, rank)
    if not fh_disable:
        fh = logging.FileHandler(os.path.join(log_path, hour_string))
        fh.setFormatter(formatter)
        fh.setLevel(fh_level)
        logger.addHandler(fh)
    
    return logger


def print_rank_0(msg, rank=0, level=logging.INFO, flush=True):
    """
    Print by single rank for multi-GPUs training

    Args:
        rank (int): Process rank for multi-GPUs training.
        level (logging.level): Level used for logger level control.
    """
    if "logger" not in globals() and rank<=0:
        global logger
        # Create logger when pring_rank_0 being used.
        logger = configure_logging(os.environ.get("LOG_FLODER", "log"), rank)
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if rank <= 0:
        logger.log(msg=msg, level=level)
        if flush:
            logger.handlers[0].flush()
            if len(logger.handlers) == 2:
                logger.handlers[1].flush()

def get_merged_state_dict(ckpt_path: str = None, 
                          partial_ckpt_path: str = None) -> dict:
    def load_state_dict(path):
        if path:
            ckpt = torch.load(path, map_location='cpu')
            return (ckpt.get('model_state_dict', ckpt),
                    ckpt.get('optimizer_state_dict', {}),
                    ckpt.get('lr_scheduler_state_dict', {}))
        return {},{},{}

    model_sd, optimizer_sd, lr_scheduler_sd  = load_state_dict(ckpt_path)
    partial_model_sd, partial_optimizer_sd, partial_lr_scheduler_sd = load_state_dict(partial_ckpt_path)
    
    model_sd.update(partial_model_sd)
    optimizer_sd.update(partial_optimizer_sd)
    lr_scheduler_sd.update(partial_lr_scheduler_sd)
    return model_sd, optimizer_sd, lr_scheduler_sd
    
def load_ckpt(model:Module, 
              ckpt_path: str = None, 
              partial_ckpt_path: str = None,
              model_sd: Optional[Dict] = None,
              rank: int = 0):
    """
    load model checkpoint safely.
    
    Args:
        model: Any pytorch model.
        ckpt_path: Path of checkpoint of all params.
        partial_ckpt_path: Path of patial model checkpoint. Must be provieded if load trainable params and pretrained params.
    """

    if model_sd is None:
        model_sd, _, _ = get_merged_state_dict(ckpt_path, partial_ckpt_path)
    
    try:
        sd_has_model_attr = any(['model.' in key for key in model_sd.keys()])
        model_has_model_attr = hasattr(model, 'model')

        if model_has_model_attr and not sd_has_model_attr:
            incompatible_keys = model.model.load_state_dict(model_sd, strict=False)
        elif not model_has_model_attr and sd_has_model_attr:
            model_sd = {k.replace('model.', ''):v for k,v in model_sd.items()}
            incompatible_keys = model.load_state_dict(model_sd, strict=False)
        else:
            incompatible_keys = model.load_state_dict(model_sd, strict=False)
            
        print_rank_0(f'--->Missing keys:{incompatible_keys.missing_keys}.', rank=rank)
        print_rank_0(f'--->Unexpected keys:{incompatible_keys.unexpected_keys}.', rank=rank)
    except Exception:
        print_rank_0(f'--->Checkpoint loading failed as error: {format_exc()} occured.', rank=rank, level=logging.ERROR)

    gc.collect()
    

def load_ckpt_for_train(model: Module, 
                        ckpt_path: str, 
                        partial_ckpt_path: str = None, 
                        rank: int = 0,
                        optimizer: torch.optim.Optimizer = None, 
                        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                        return_model_sd: bool = True,
                        return_optimizer_sd: bool = True,
                        return_lr_scheduler_sd: bool = True):

    model_sd, optimizer_sd, lr_scheduler_sd = get_merged_state_dict(ckpt_path, partial_ckpt_path)
    
    if optimizer:
        optimizer.load_state_dict(optimizer_sd)  
        optimizer_sd = {}
    if lr_scheduler:
        lr_scheduler.load_state_dict(lr_scheduler_sd)
        lr_scheduler_sd = {}
    if model:
        load_ckpt(model=model, model_sd=model_sd, rank=rank)
        model_sd = {}

    if not return_model_sd and model_sd:
        model_sd = {}
    if not return_optimizer_sd and optimizer_sd:
        optimizer_sd = {}
    if not return_lr_scheduler_sd and lr_scheduler_sd:
        lr_scheduler_sd = {}
    gc.collect()
    return model_sd, optimizer_sd, lr_scheduler_sd

STR_DTYPE_TO_TORCH_DTYPE = immutabledict.immutabledict({
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
    'bf16': torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32
})

@contextlib.contextmanager
def set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)
