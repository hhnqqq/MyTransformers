import os
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.utils import is_liger_kernel_available
if is_liger_kernel_available():
    from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM

from model import *
from common.lora_modules import *
from common.registry import registry
from common.utils.params_manager import set_up_multimodal_config
from common.utils import (
    load_ckpt_for_train,
    print_rank_0, read_config, load_ckpt,
    dict_to_dataclass, set_default_tensor_type, STR_DTYPE_TO_TORCH_DTYPE)

def load_huggingface_model_config(args):
    model_name_or_path = args.model_name_or_path
    if os.path.exists(model_name_or_path):
        config_path = os.path.join(model_name_or_path, 'config.json')
        # Especially, write lora layers config in the config file is needed.
        model_config = read_config(config_path)
        # Convert dict config to dataclass.
        model_config = dict_to_dataclass('model_config', model_config)
    else:
        model_config = None
    return model_config        

def load_huggingface_model(args):
    # Train with huggingface pretrained model. Only support data parallel training.
    return_dataset_kwargs = {}
    print_rank_0(f'--->Using tokenizer and model from huggingface with path: {args.model_name_or_path}', args.global_rank)
    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
                                                 trust_remote_code=True,
                                                 torch_dtype=STR_DTYPE_TO_TORCH_DTYPE[args.default_dtype],
                                                 attn_implementation="sdpa",
                                                 device_map=f"cuda:{args.local_rank}",
                                                 use_cache=False if args.activation_checkpoint else True)
    if args.activation_checkpoint:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # The main checkpoint is loaded by `from_pretrained` function. Partial checkpoint is used for LoRA.
    if args.partial_ckpt_path:
        load_ckpt(model=model, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
    model_config = load_huggingface_model_config(args)
    if args.num_pp_stages:
        print_rank_0(f'Pipeline parallelism is not supported for huggingface models, ignore.')
    if args.num_sp_stages:
        print_rank_0(f'Sequence parallelism is not supported for huggingface models, ignore.')

    # For compatibility with Dataset classes.
    tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id = tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id
    tokenizer.label_pad_id = -100
    return model, tokenizer, model_config, return_dataset_kwargs

def load_local_model(args):
    return_dataset_kwargs = {}

    # Train with local defined model.
    print_rank_0(f'--->Using tokenizer: {args.tokenizer_name} with path: {args.tokenizer_path}', args.global_rank)
    tokenizer = registry.get_tokenizer_class(args.tokenizer_name)(args.tokenizer_path)
    config_type = '_'.join([args.model_name, args.variant])
    model_config = registry.get_model_config_class(config_type)()
    print_rank_0(f'--->Using model config: {config_type}', args.global_rank)
    model_config.vocab_size = tokenizer.n_words
    # Load model in default dtype to avoid OOM (cpu memory).
    with set_default_tensor_type(args.default_dtype):
        model = registry.get_model_class(args.model_name)(model_config)

    print_rank_0(f'--->Using model: {args.model_name}, and loading its trainning variant', args.global_rank)

    # Load checkpoint if checkpoint path is provieded.
    if args.ckpt_path is not None and args.from_pretrained:
        if args.model_name == 'gemma':
            _, optimizer_sd, lr_scheduler_sd = load_ckpt_for_train(model=model, ckpt_path=args.ckpt_path, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
        else:
            _, optimizer_sd, lr_scheduler_sd = load_ckpt_for_train(model=model.model, ckpt_path=args.ckpt_path, partial_ckpt_path=args.partial_ckpt_path, rank=args.global_rank)
        print_rank_0(f'--->Using pretrained checkpoint at {args.ckpt_path}', args.global_rank)
        model_config.optimizer_sd = optimizer_sd
        model_config.lr_scheduler_sd = lr_scheduler_sd

        if args.multimodal and hasattr(model, 'multimodal_model') and args.multimodal_model_ckpt_path:
            set_up_multimodal_config(model_config, args)
            return_dataset_kwargs['multimodal_k_tokens'] = args.multimodal_k_tokens
            load_ckpt(model=model.multimodal_model, ckpt_path=args.multimodal_model_ckpt_path, rank=args.global_rank)
            print_rank_0(f'--->Using pretrained multimodal model checkpoint at {args.multimodal_model_ckpt_path}', args.global_rank)
            multimodal_tokenizer = registry.get_tokenizer_class(args.multimodal_tokenizer_name)(args.multimodel_tokenizer_path)
            return_dataset_kwargs['multimodal_tokenizer'] = multimodal_tokenizer
    else:
        print_rank_0('--->Not using pretrained checkpoint to start traning.', args.global_rank)

    # Load config from model config to argument parser namespace.
    args.head_dim = model_config.head_dim
    args.head_num = model_config.n_heads
    args.hidden_size = model_config.dim
    args.num_layers = model_config.n_layers
    args.rope_theta = model_config.rope_theta if args.rope_theta is None else args.rope_theta
    args.pad_id = tokenizer.pad_id

    # Convert model to trainable model for given training type.
    if args.num_pp_stages:
        pipe_model_cls = registry.get_pipeline_model_class(args.model_name)
        model = pipe_model_cls(model, args)
    else:
        train_model_cls = registry.get_train_model_class(args.model_name)
        model = train_model_cls(model, args)
        
    # Convert dtype to avoid inconsistency between default dtype and checkpoint dtype.
    torch.cuda.empty_cache()
    gc.collect()
    model.to(STR_DTYPE_TO_TORCH_DTYPE[args.default_dtype]).to(args.device)
    return model, tokenizer, model_config, return_dataset_kwargs

def load_model(args):
    """
    Load model from args.
    If `args.huggingface`, then the model and tokenizer will be loaded from transformers.
    Else the model and tokenizer will be load from MyTransformers.

    Note that huggingface models do not support pipeline parallelism and sequence parallelism for now.

    It is possisble to support sequence parallelism by replace the Attention class of huggingface models 
    using a monkey patch.
    """
    if args.huggingface:
        return load_huggingface_model(args)
    else:
        return load_local_model(args)