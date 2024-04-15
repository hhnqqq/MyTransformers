import os
import time
import json
import pytz
import torch
import random
import logging
import deepspeed
import configparser
import numpy as np
from datetime import datetime
import torch.distributed as dist
import common.utils.parallel_states as parallel_states

"""with 环境用法：进入with语句之后，类中__enter__对应的返回值将被赋值给as后面的变量名
with之后的代码执行"""
class Timer(object):
    def __init__(self, start=None, n_round=2):
        self.round = n_round
        self.start = round(start if start is not None else time.time(), self.round)
        self.loop_start = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop = round(time.time(), self.round)
        self.time_cost = round(self.stop - self.start, self.round)
        self.formatted_time_cost = f"{int(self.time_cost//60)}min {self.time_cost%60}s"
        return exc_type is None

    def average_time(self, entry):
        current_time = round(time.time(), self.round)
        if entry=='start':
            if self.loop_start is None:
                self.loop_start = current_time
        elif entry=='end':
            loop_end = current_time
            self.loop_time = round(loop_end - self.loop_start, self.round)
            self.loop_start = None
        else:
            raise ValueError("Invalid entry value. Expected 'start' or 'end'.")

def load_ckpt(model, ckpt_path):
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def read_config(file_path, encoding='utf-8'):
    """读取配置文件"""
    if file_path.endswith('.json'):
        with open(file_path, 'r', encoding=encoding) as f:
            config = json.load(f)
    elif file_path.endswith('.ini'):
        config = configparser.ConfigParser()
        config.read(file_path)
    else:
        raise ValueError("无法读取不支持的文件格式")
    return config

def ensure_directory_exists(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f'---> Directory:{directory} is not existed. created a new floder')

def count_trainable_parameters(model):
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    num_trainable_params = sum([torch.numel(p) for p in trainable_params])
    return num_trainable_params

def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

class DataCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_id

    def __call__(self, examples):
        input_ids_list, labels_list = [], []
        for instance in examples:
            input_ids = instance["input_ids"]
            labels = instance["labels"]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {"input_ids": torch.stack(input_ids_list),
                "labels": torch.stack(labels_list)}
    
def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output

def set_random_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

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
    if args.num_sp_stages is not None:
        assert args.atten_type == 'ulysses_atten', 'when using sequence parallism, the attention type must be `ulysses_atten`'
        parallel_states.initialize_model_parallel(sequence_model_parallel_size=args.num_sp_stages)
    else:
        parallel_states.initialize_model_parallel()
    return device, args

# ------------------logging----------------------------
def configure_logging(log_path):
    sh_level = os.environ.get("PRINTLEVEL", logging.DEBUG)
    fh_level = os.environ.get("LOGLEVEL", logging.INFO)
    logger = logging.getLogger("DNALLaMa")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    sh.setLevel(sh_level)
    logger.addHandler(sh)

    timezone = pytz.timezone('Asia/Shanghai')
    date_string, hour_string = datetime.now().strftime('%y-%m-%d'), datetime.now().strftime('%H-%M')+'.log'
    log_path = os.path.join(log_path, date_string)
    ensure_directory_exists(log_path)
    fh = logging.FileHandler(os.path.join(log_path, hour_string))
    fh.setFormatter(formatter)
    fh.setLevel(fh_level)
    logger.addHandler(fh)
    return logger

logger = configure_logging(os.environ.get("LOG_FLODER", "log"))

def print_rank_0(msg, rank=0, level=logging.INFO, flush=True):
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    if rank <= 0:
        logger.log(msg=msg, level=level)
        if flush:
            logger.handlers[0].flush()