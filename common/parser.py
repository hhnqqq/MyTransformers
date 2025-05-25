import os
import json
import inspect
import argparse
import deepspeed
from typing import Union, List
from transformers import logging as transformers_logging

from common.utils import print_rank_0, init_dist, set_random_seed

def base_parser():
    parser = argparse.ArgumentParser()
    # ------------------- Path -------------------------
    parser.add_argument('--train-dataset-path',type=str, default=None,
                        help='The path of training dataset file (such as /datasets/gsm8k.jsonl).')
    parser.add_argument('--eval-dataset-path',type=str, default=None,
                        help='The path of evaluating dataset file.')
    parser.add_argument('--ckpt-path',type=str, default=None,
                        help='The path of model checkpoint (such as /models/llama3.pth).')
    parser.add_argument('--tokenizer-path',type=str, default=None,
                        help='The path of tokenizer checkpoint (such as /tokenizers/llama3.model)')
    parser.add_argument('--output-path',type=str,
                        help='The path of output floder.')
    parser.add_argument('--tokenizer-name',type=str, default=None,
                        help='The name of tokenizer, can used to acquire tokenizer_path')
    parser.add_argument('--model-name',type=str, default=None,
                        help='The name of model, can used to acquire model_path (local model) or model_name_or_path (huggingface)')
    parser.add_argument('--model-name-or-path', type=str, default=None,
                        help='`model_name_or_path` is used to init huggingface model and tokenizer. Please refer to train.load_model.load_huggingface_model.')
    parser.add_argument('--train-dataset-name',type=str, default=None,
                        help='Name of dataset file, use this if you have registered dataset in paths.json')
    parser.add_argument('--eval-dataset-name',type=str, default=None)
    parser.add_argument('--partial-ckpt-path',type=str, default=None, 
                        help='This argument is useful when train model base on previous trainable params from previous experiment.')
    parser.add_argument('--dataset-class-name', type=str, default='iterable',
                        help='Name of dataset class, please refer to dataset_classes for specific classes.')

    # --------------------- Logging -----------------------
    parser.add_argument('--tensorboard', action='store_true',
                        help='Set this to enable tensorboard logging.')
    parser.add_argument('--tb-log-dir', type=str, default=None,
                        help='Path of tensorboard log dir')
    parser.add_argument('--wandb', action='store_true',
                        help='Set this to enable wandb logging.')
    parser.add_argument('--wandb-team', type=str, default=None,
                        help='Team of wandb.')
    parser.add_argument('--wandb-cache-dir', type=str, default=None,
                        help='Cache dir of wandb')
    parser.add_argument('--wandb-dir', type=str, default=None,
                        help='Dir of wandb')
    parser.add_argument('--test-code', action='store_true', help='add this argument to avoid creating log file.')
    parser.add_argument('--profile-log-dir', type=str, default=None,   
                        help='Path of profiler log dir')

    return parser

def train_parser(parser):
    group = parser.add_argument_group('train', 'training configurations')

    # --------------- Core hyper-parameters --------------- 
    group.add_argument('--experiment-name', type=str, default='MyModel', 
                       help='The name of the experiment for summary and checkpoint.')
    group.add_argument('--train-iters', type=int, default=None, 
                       help='Total number of iterations to train over all training runs')
    group.add_argument('--epochs', type=int, default=None, 
                       help='Number of training epochs')
    group.add_argument('--fp16', action='store_true', 
                       help='Run the model in fp16 mode')
    group.add_argument('--bf16', action='store_true', 
                       help='Run the model in bf16 mode')
    group.add_argument('--variant', type=str, default='2b', 
                       help='The variant of the model.')
    group.add_argument('--save-interval', type=int, default=None, 
                       help='Number of iterations between saves')
    group.add_argument('--save-epoch', type=int, default=None, 
                       help='Number of epochs between saves')
    group.add_argument('--eval-interval', type=int, default=5000, 
                       help='Number of iterations between evaluations')
    group.add_argument('--device', type=str, default='cuda', 
                       help='The device to load the model')
    group.add_argument('--mode', type=str, default='pretrain', choices=['pretrain', 'sft', 'dual_rl', 'rlhf'],
                       help='The training mode')
    group.add_argument('--from-pretrained', action='store_true',
                       help='Train the model from a pretrained checkpoint')
    group.add_argument('--batch-size-per-gpu', type=int, default=4, 
                       help='Batch size on a single GPU. batch-size * world_size = total batch_size.')
    
    # --------------------------- parameters ----------------------------
    group.add_argument('--enable-list', nargs='+', type=str, default=None,
                       help='List of enabled parameters')
    group.add_argument('--disable-list', nargs='+', type=str, default=None,
                       help='List of disabled parameters')
    group.add_argument('--activation-checkpoint', action='store_true',
                       help='Train the model with activation checkpoint')
    group.add_argument('--params-to-save', nargs='+', type=str, default=None,
                       help='Params need to be saved even if they do not need gradients.')
    
    # -------------------------- others ----------------------------
    group.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    group.add_argument('--show-loss-step', type=int, default=1)
    group.add_argument('--show-avg-loss-step', type=int, default=10)
    group.add_argument('--rope-theta', default=None,
                       help='Rope theta')
    group.add_argument('--train-pi', type=int, default=None,
                       help='The interpolation factor of RoPE, which is used to enhance the sequence length\
                        In the case of a non-existent interpolation multiple, the rope will remain in its original state.')
    group.add_argument('--atten-type', type=str, default="",
                       help='Type of attention. For example: flash-atten')
    group.add_argument('--loss-fct', type=str, default='ce')
    group.add_argument('--fuse-linear-loss', action='store_true')

    return parser

def optimizer_parser(parser):
    group = parser.add_argument_group('optimizer', 'optimizer configurations')

    # --------------------- optimizer -----------------------
    group.add_argument('--diy-optimizer', action='store_true', 
                       help='Whether to DIY the optimizer. '
                       'DeepSpeed optimizer will be used as default optimizer.')
    group.add_argument('--disable-zero-optimizer', action='store_true')
    group.add_argument('--offload-optimizer', action='store_true')
    group.add_argument('--eval-batch-size-per-gpu', type=int, default=4, 
                       help='Evaluation batch size on a single GPU. batch-size * world_size = total batch_size.')
    group.add_argument('--lr', type=float, default=1.0e-4, 
                       help='Initial learning rate')
    group.add_argument('--eps', type=float, default=1e-8, 
                       help='Initial epsilon for the optimizer')
    group.add_argument('--betas', nargs='+', type=float, default=[0.9,0.999], 
                       help='Initial beta values for the optimizer')
    group.add_argument('--warmup-min-lr', type=float, default=1.0e-5, 
                       help='Minimum learning rate for deepspeed warmup configurations')
    group.add_argument('--warmup-max-lr', type=float, default=2.0e-4, 
                       help='Maximum learning rate for deepspeed warmup configurations')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1, 
                       help='Run optimizer after every gradient-accumulation-steps backwards')
    group.add_argument('--auto-warmup-steps', type=int, default=10, 
                       help='The fixed warmup steps for training')
    group.add_argument('--auto-warmup-rate', type=float, default=0.05, 
                       help='The warmup rate for fixed warmup steps')
    group.add_argument('--warmup', type=float, default=0.01, 
                       help='Percentage of data to warm up on (.01 = 1% of all training iters). Default 0.01')
    group.add_argument('--weight-decay', type=float, default=5e-4, 
                       help='Weight decay coefficient for L2 regularization')
    group.add_argument('--lr-decay-style', type=str, default='cosine', choices=['constant', 'linear', 'cosine', 'exponential', 'cosine_restarts'], 
                       help='Learning rate decay function')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--lr-decay-iters', type=int, default=None, 
                       help='Number of iterations to decay LR over. If None, defaults to --train-iters * --epochs')
    group.add_argument('--optim-type', type=str, default=None, 
                       help='Type of the optimizer, this arg will be useful when diy-optimizer is true')
    group.add_argument('--clip-grad-max-norm', type=float, default=1.0,
                       help='Threshold norm value for gradient')
    
    return parser

def dataset_parser(parser):
    group = parser.add_argument_group('dataset', 'dataset configurations')

    # ---------------------------- dataset ------------------------------
    group.add_argument('--skip-eval', action='store_true')
    group.add_argument('--dataset-input-field', type=str, default='input',
                       help='Input column of the dataset.')
    group.add_argument('--dataset-output-field', type=str, default='output',
                       help='Ouput column of the dataset.')
    group.add_argument('--read-nums', type=int, default=None,
                       help='The number of data to read. If this value is None, the dataset will read all data')
    group.add_argument('--eval-read-nums', type=int, default=None,
                       help='The number of evaluation data to read. If this value is None, the dataset will read all data')
    group.add_argument('--max-len', type=int, default=None,
                       help='Maximum length of tokens for a single data sample')
    group.add_argument('--max-src-len', type=int, default=None,
                       help='Maximum length of input tokens')
    group.add_argument('--eval-max-len', type=int, default=None,
                       help='Maximum evaluation length of tokens for a single data sample')
    group.add_argument('--eval-max-src-len', type=int, default=None,
                       help='Maximum evaluation length of input tokens')
    group.add_argument('--meta-prompt', type=Union[str,List[str]], default=None,
                       help='The systematic prompt for the input')
    group.add_argument('--prefix', type=Union[str,List[str]], default=None,
                       help='The prefix added to the input')
    group.add_argument('--postfix', type=Union[str,List[str]], default=None,
                       help='The postfix added to the input')
    group.add_argument('--prompt-path', type=str, default=None)
    group.add_argument('--batching-stretegy', type=str, default='padding', choices=['padding', 'packing'],
                       help='The stretegy for batching dataset')
    group.add_argument('--dataset-weights', type=int, nargs='+', default=None)
    group.add_argument('--read-start-step', type=int, default=None)
    
    return parser

def peft_parser(parser):
    group = parser.add_argument_group('peft', 'parameter efficient training configurations')

    # --------------------------- lora ----------------------------------
    group.add_argument('--use-lora', action='store_true',
                       help='Whether to use LoRA')
    group.add_argument('--use-vera', action='store_true', default=None,
                       help='Whether to use vera')
    group.add_argument('--lambda-b-init-method', type=str, default=None,
                       help='Init method for lora lambda b')
    group.add_argument('--lambda-d-init-method', type=str, default=None,
                       help='Init method for lora lambda b')
    group.add_argument('--use-lora-pro', action='store_true',
                       help='Whether to use LoRA-Pro optimizer')
    group.add_argument('--use-dora', action='store_true',
                       help='Whether to use DoRA')
    group.add_argument('--lora-scaler', type=int, default=32,
                       help='Scaler factor for lora')
    group.add_argument('--use-lora-plus', action='store_true',
                       help='Whether to use LoRA+')
    group.add_argument('--use-mos-lora', action='store_true',
                       help='Whether to use mos lora')
    group.add_argument('--use-me-lora', action='store_true',
                       help='Whether to use me lora')
    group.add_argument('--me-lora-n-split', type=int, default=2)
    group.add_argument('--me-lora-usage',type=str, default='compress', choices=['compress','higher_rank'])
    group.add_argument('--me-lora-forward-method', type=str, default='for', choices=['for','einsum'])
    group.add_argument('--lora-fa', action='store_true',
                       help='Whether to use LoRA FA')
    group.add_argument('--use-rslora', action='store_true',
                       help='Whether to use rslora')
    group.add_argument('--use-gora', action='store_true',
                       help='Whether to use gora')
    group.add_argument('--gora-init-method', type=str, default='weight_svd')
    group.add_argument('--use-pissa', action='store_true',
                       help='Whether to use pissa')
    group.add_argument('--use-milora', action='store_true',
                       help='Whether to use milora')
    group.add_argument('--use-olora', action='store_true',
                       help='Whether to use olora')
    group.add_argument('--use-delta-lora', action='store_true',
                       help='Whether to use delta-lora')
    group.add_argument('--use-mora', action='store_true',
                       help='Whether to use mora.')
    group.add_argument('--use-mola', action='store_true',
                       help='Whether to use mola.')
    group.add_argument('--mola-type', type=str, default='triangle')
    group.add_argument('--mora-type', type=str, default='rope')
    group.add_argument('--delta-lora-start-steps', type=int, default=500,
                       help='Start to compute delta lora weights')
    group.add_argument('--delta-lora-update-ratio', type=int, default=2)
    group.add_argument('--pissa-n-iters', type=int, default=1, 
                       help='The number of iterations determines the trade-off \
                        between the error and computation time')
    group.add_argument('--pissa-keep-init-weights', action='store_true')
    group.add_argument('--milora-n-iters', type=int, default=1, 
                       help='The number of iterations determines the trade-off \
                        between the error and computation time')
    group.add_argument('--lora-rank', type=int, default=8,
                       help='The rank of LoRA')
    group.add_argument('--lora-plus-scaler', type=int, default=16,
                       help='The scaler of learning rate of LoRA weight b \
                       In the default case, the learning rate of weight b is 16 times of a')
    group.add_argument('--replace-modules', type=str, nargs='+', default=None,
                       help='List of modules to be replaced by LoRA')
    group.add_argument('--weight-a-init-method', type=str, default=None,
                       help='Init method for lora weight a')
    group.add_argument('--weight-b-init-method', type=str, default=None,                       
                       help='Init method for lora weight b')
    group.add_argument('--weight-ab-mixer-init-method', type=str, default=None,
                       help='Init method for lora weight ab mixer')
    group.add_argument('--use-lora-ga', action='store_true',
                       help='Wheather to use lora ga')
    group.add_argument('--lora-ga-n-steps', type=int, default=8,
                       help='N steps for lora-ga to estimate full-rank gradient.')
    group.add_argument('--gora-n-steps', type=int, default=8,
                       help='N steps for gora to estimate full-rank gradient.')
    group.add_argument('--gora-max-rank', type=int, default=9999)
    group.add_argument('--gora-min-rank', type=int, default=1)
    group.add_argument('--gora-softmax-importance', action='store_true')
    group.add_argument('--gora-scale-by-lr', action='store_true')
    group.add_argument('--gora-lr', type=float, default=1e-3)
    group.add_argument('--gora-features-func', type=str, default=None)
    group.add_argument('--gora-temperature', type=int, default=0.5)
    group.add_argument('--gora-rank-stablize', action='store_true')
    group.add_argument('--gora-dynamic-scaling', action='store_true')
    group.add_argument('--gora-allocate-stretagy', type=str, default='moderate')
    group.add_argument('--gora-scale-importance', action='store_true')
    group.add_argument('--gora-importance-type', type=str, default='union_frobenius_norm')
    group.add_argument('--gora-stable-gemma', type=float, default=0.02)
    group.add_argument('--lora-ga-scale-method', type=str, default='gd')
    group.add_argument('--lora-ga-reset-weight', action='store_true',
                       help='Whether to reset pretrained weight when using LoRA-GA, this will improve numerical stability.')
    group.add_argument('--relora-steps', type=int, default=None,
                       help='How much step to merge and reset the lora weight')
    group.add_argument('--relora-warmup-steps', type=int, default=None)
    group.add_argument('--relora-counts', type=int, default=None)
    group.add_argument('--relora-reset-optimizer', action='store_true')
    group.add_argument('--relora-fully-reset-optimizer', action='store_true')
    group.add_argument('--relora-optimizer-random-pruning', type=float, default=None)
    group.add_argument('--relora-optimizer-magnitude-pruning', type=float, default=None)
    group.add_argument('--use-plora', action='store_true')
    group.add_argument('--plora-momentum', type=float, default=0.1)
    group.add_argument('--use-lora-moe', action='store_true')
    group.add_argument('--lora-moe-n-experts', type=int, default=2)
    group.add_argument('--lora-moe-top-k', type=int, default=2)
    group.add_argument('--lora-dropout', type=float, default=None,
                       help='The dropout rate for lora weight.')
    group.add_argument('--run-lora-in-fp32', action='store_true',
                       help='Whether to keep lora weight in fp32.')
    
    # --------------------------- galore ----------------------------------
    group.add_argument('--use-galore', action='store_true',
                       help='Whether to use Galore')
    group.add_argument('--galore-rank', type=int, default=8,
                       help='The rank of Galore')
    group.add_argument('--galore-scaler', type=float, default=0.25,
                       help='The scaler of Galore')
    group.add_argument('--galore-per-layer', action='store_true')


    # --------------------------- adalora ----------------------------------
    group.add_argument('--use-adalora', action='store_true',
                       help='Whether to use adalora')
    group.add_argument('--target-r', type=int, default=16,
                       help='Target Lora matrix dimension.')
    group.add_argument('--init-r', type=int, default=8,
                       help='Initial Lora matrix dimension.')
    group.add_argument('--tinit', type=int, default=0,
                       help='The steps of initial warmup.')
    group.add_argument('--tfinal', type=int, default=0,
                       help='The steps of final warmup.')
    group.add_argument('--deltaT', type=int, default=1,
                       help='Step interval of rank allocation.')
    group.add_argument('--beta1', type=float, default=0.85,
                       help='Hyperparameter of EMA.')
    group.add_argument('--beta2', type=float, default=0.85,
                       help='Hyperparameter of EMA.')
    group.add_argument('--orth-reg-weight', type=float, default=0.5,
                       help='The orthogonal regularization coefficient.')

    # --------------------------- increlora ----------------------------------
    group.add_argument('--use-increlora', action='store_true',
                       help='Whether to use increlora')
    group.add_argument('--top-h', type=int, default=2,
                       help='The number of selected modules per allocation.')

    return parser

def multimodal_parser(parser):
    group = parser.add_argument_group('multimodal', 'Multimodal encoder-decoder architecture training configurations')

    # ----------------------- multimodal -----------------------------
    group.add_argument('--multimodal', action='store_true',
                       help='Weather to use multimodal model')
    group.add_argument('--multimodal-model-ckpt-path', type=str, default=None,
                       help='Path of the checkpoint of multimodal model(Vision encoder or DNA encoder....)')
    group.add_argument('--multimodal-tokenizer-name', type=str, default=None,
                       help='Name of multimodal tokenizer')
    group.add_argument('--multimodal-projector-type', type=str, default='linear', choices=['mlp', 'linear', 'qformer', 'resampler'])
    group.add_argument('--multimodal-projector-layers', type=int, default=1)
    group.add_argument('--multimodal-k-tokens', type=int, default=32)
    group.add_argument('--multimodal-sample-mode', type=str, default='pool')
    group.add_argument('--multimodal-encode-fp32', action='store_true')

    return parser

def ds_parser(parser):
    group = parser.add_argument_group('ds', 'ds configurations')
    group.add_argument('--ds-config-path',type=str,
                      help='path of ds configuration file',)
    group.add_argument('--local_rank',type=int,default=-1,
                      help='local rank passed from distributed launcher',)
    group.add_argument('--global-rank', default=-1, type=int, 
                      help='global rank')
    group.add_argument('--with-aml-log', default=True, 
                      help='Use Azure ML metric logging. This argument is not enabled currently')
    group.add_argument('--offload-param', action='store_true')
    group.add_argument('--huggingface', action='store_true')
    group.add_argument('--csv-monitor', action='store_true')
    group.add_argument('--monitor-file-path', type=str)
    group.add_argument('--num-pp-stages', type=int, default=None,
                       help='the pipeline stages, this value must be divisible by your GPU num')
    group.add_argument('--num-sp-stages', type=int, default=None,
                       help='the sequence parallel stages, this value must be divisible by your GPU num')
    group.add_argument('--save-trainable', action='store_true')
    group.add_argument('--encode-single-gene', action='store_true')
    group.add_argument('--all-reduce-loss', action='store_true')
    group.add_argument('--zero-stage', type=int, default=-1)

    # Include DeepSpeed configuration arguments
    parser = deepspeed.add_config_arguments(parser)
    return parser

def get_args():
    """
    Parse arguments from ArgumentParser and check if the arguments are legeal.
    """
    parser = base_parser()
    parser = dataset_parser(parser)
    parser = train_parser(parser)
    parser = peft_parser(parser)
    parser = multimodal_parser(parser)
    parser = optimizer_parser(parser)
    parser = ds_parser(parser)

    args = parser.parse_args()
    os.environ['EXPERIMENT_NAME'] = args.experiment_name
    args = init_dist(args)

    set_random_seed(args.seed)
    print_rank_0(f'--->Using random seed: {args.seed}', args.global_rank)
    # Log transformers related information only on rank 0.
    if args.global_rank != 0:
        transformers_logging.set_verbosity_error()
    else:
        transformers_logging.set_verbosity_info()
        
    if args.tensorboard and args.tb_log_dir is None:
        raise ValueError("`tb-log-dir` need to be set if tensorboard logging is needed")
    if args.wandb and (args.wandb_dir is None or args.wandb_cache_dir is None):
        raise ValueError("`wandb-dir` and `wandb-cache_dir` need to be set if wandb logging is needed")
    if (args.save_interval or args.save_epoch) and args.output_path is None:
        raise ValueError("Output path can not be None when model saving is required.")
    if args.fp16 and args.bf16:
        raise ValueError("Cannot specify both fp16 and bf16.")
    if args.train_iters is not None and args.epochs is not None:
        raise ValueError('Only one of train_iters and epochs should be set.')
    if args.multimodal:
        if args.multimodal_projector_type == 'mlp':
            if not args.multimodal_projector_layers > 1:
                raise ValueError('Mlp module layer count must greater than 1')
    
    if args.fp16:
        args.default_dtype = 'fp16'
    elif args.bf16:
        args.default_dtype = 'bf16'
    else:
        args.default_dtype = 'fp32'

    if args.train_iters is None and args.epochs is None:
        args.train_iters = 10000 # default 10k iters
        print_rank_0('No train_iters (recommended) or epochs specified, use default 10k iters.', level='WARNING', rank=args.global_rank)

    if args.zero_stage > 0 and not args.fp16 and not args.bf16:
        print_rank_0('Automatically set fp16=True to use ZeRO.', args.global_rank)     
        args.fp16 = True
        args.bf16 = False
    
    mt_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(lambda: None))))
    if args.ds_config_path is None and args.zero_stage > 0:
        args.ds_config_path = os.path.join(mt_dir, "ds_config", f"zero{args.zero_stage}_config.json")
    print_rank_0(f"--->Using {args.ds_config_path} as deepspeed config path", args.global_rank)
        
    if args.params_to_save is not None:
        if isinstance(args.params_to_save, str):
            args.params_to_save = args.params_to_save.split('_')
        print_rank_0(f"--->Parameters to save: {args.params_to_save}", args.global_rank)

    if args.prompt_path:
        prompt_info = json.load(open(args.prompt_path, 'r'))
        args.meta_prompt = prompt_info['meta_prompt']
        args.prefix = prompt_info['prefix']
        args.postfix = prompt_info['postfix']
    
    # If args.test_code, the log file and tb writer will not be created.
    if args.test_code:
        os.environ['NO_LOG_FILE'] = 'true'
    return args

def overwrite_args_by_dict(args, overwrite_args={}):
    for k in overwrite_args:
        setattr(args, k, overwrite_args[k])

def update_args_with_file(args, path):
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # expand relative path
    folder = os.path.dirname(path)
    for k in config:
        # all the relative paths in config are based on the folder
        if k.endswith('_path'): 
            config[k] = os.path.join(folder, config[k])
            print_rank_0(f'> parsing relative path {k} in model_config as {config[k]}.', args.global_rank)
    args = vars(args)
    for k in list(args.keys()):
        if k in config:
            del args[k]
    args = argparse.Namespace(**config, **args)
    return args
