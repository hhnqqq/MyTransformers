# TODO:利用多张显卡跑测试集
import os
import json
import argparse

import torch
import pandas as pd

from functools import partial

from model import *
from common.registry import registry
from common.lora_modules import LinearWithLoRAGA, LinearWithGoRA
from common.utils import set_random_seed, load_ckpt
from common.utils.utils import set_default_tensor_type
from common.utils import parallel_states as parallel_states

from lora_set_up import switch_to_lora
from me_tdlora import LinearWithMETDLoRA
from me_tdlora_monarch import LinearWithMETDMONARCHLoRA
from me_tdlora_mixer import LinearWithMETDMIXERLoRA
from me_tdlora_dynamic_n import LinearWithMETDDYNAMICNLoRA
from me_tdlora_compress import LinearWithMETDCOMPRESSLoRA
from me_tdlora_similarity import LinearWithMETDSIMILARITYLoRA
from lora_ga_pro import LinearWithLoRAGAPro

from tqdm import tqdm
from argparse import Namespace

format_dict = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "json": pd.read_json,
    "jsonl": partial(pd.read_json, lines=True)
}

INPUT_PROMPT='''===>Please enter your prompt
Enter quit() to quit
Enter output_len() to change output_len
Enter prefix() to change prefix
Enter meta_prompt() to change meta prompt
Enter clear() to clear meta prompt and prefix:'''

def stratified_sample(df, strata_col, total_sample_size, equal_sample=False, replace=False):
    strata_sizes = df[strata_col].value_counts()
    
    if equal_sample:
        sample_size_per_strata = total_sample_size // len(strata_sizes)
        remaining_samples = total_sample_size % len(strata_sizes)
        
        sample_sizes = {level: sample_size_per_strata for level in strata_sizes.index}
        for _ in range(remaining_samples):
            sample_sizes[next(iter(sample_sizes))] += 1
    else:
        total_size = df.shape[0]
        strata_proportions = strata_sizes / total_size
        
        sample_sizes = {level: int(proportion * total_sample_size) for level, proportion in strata_proportions.items()}
        
        remaining_samples = total_sample_size - sum(sample_sizes.values())
        for level in sorted(sample_sizes, key=lambda x: -sample_sizes[x])[:remaining_samples]:
            sample_sizes[level] += 1
    sampled_df = pd.concat([
        df[df[strata_col] == level].sample(n=sample_sizes[level], replace=replace)
        for level in sample_sizes.keys()
    ])
    
    return sampled_df.sample(frac=1).reset_index()


def main(args):
    # Construct the model config.
    if args.ckpt:
        floder_path = os.path.dirname(args.ckpt)
    else:
        floder_path = os.path.dirname(args.pretrained_ckpt)
    if os.path.exists(floder_path) and os.path.isdir(floder_path):
        config_path = os.path.join(floder_path, 'config.json')
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        training_config['me_lora_forward_method'] = args.me_lora_forward_method
        training_config = Namespace(**training_config)
    
    else:
        training_config = None
    model_config = registry.get_model_config_class('_'.join([args.model_name, args.variant]))()
    model: nn.Module = registry.get_model_class(args.model_name)
    if training_config:
        if training_config.bf16:
            model_config.dtype = "bf16" 
        elif training_config.fp16:
            model_config.dtype = "fp16"
    else:
        model_config.dtype = "fp32"
    model_config.quant = args.quant
    # model_config.tokenizer = args.tokenizer
    model_config.tokenizer = args.tokenizer if args.tokenizer is not None else training_config.tokenizer_path 
    print(model_config.tokenizer)    

    # Seed random.
    set_random_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    print("Start loading model")
    with set_default_tensor_type(model_config.get_dtype()):
        model = model(model_config)
        print("load checkpoint")
        if args.pretrained_ckpt is not None:
            load_ckpt(model=model.model, ckpt_path=args.pretrained_ckpt)
            print(f"loaded pretrained weight at{args.pretrained_ckpt}")
        if training_config.use_lora:
            print("Replacing model with lora layers")
            switch_to_lora(model, 
            args=training_config)
            if ((hasattr(training_config, 'use_me_tdlora')) and training_config.use_me_tdlora) or \
                ((hasattr(training_config, 'use_me_td_monarch_lora')) and training_config.use_me_td_monarch_lora) or \
                ((hasattr(training_config, 'use_me_td_mixer_lora')) and training_config.use_me_td_mixer_lora) :
                rank_config_file = os.path.join(os.path.dirname(args.ckpt), 'rank.json')
                rank_config = json.load(open(rank_config_file,'r'))
                for name,module in model.model.named_modules():
                    if isinstance(module, LinearWithMETDLoRA) or isinstance(module, LinearWithMETDMONARCHLoRA) or isinstance(module, LinearWithMETDMIXERLoRA):
                        rank = rank_config[name]
                        module.init_method = 'vanilla'
                        # module.melora_n_split = training_config.me_n_split
                        module.dynamic_init(training_config.lora_rank, rank)
            if (hasattr(training_config, 'use_me_td_dynamic_n_lora') and training_config.use_me_td_dynamic_n_lora) or \
                ((hasattr(training_config, 'use_me_td_lora_compress')) and training_config.use_me_td_lora_compress) or \
                    ((hasattr(training_config, 'use_me_td_lora_similarity')) and training_config.use_me_td_lora_similarity):
                rank_config_file = os.path.join(os.path.dirname(args.ckpt), 'rank.json')
                rank_config = json.load(open(rank_config_file,'r'))
                if training_config.dynamic_n_allocation:
                    n_splits_file = os.path.join(os.path.dirname(args.ckpt), 'n_splits.json')
                    n_splits_config = json.load(open(n_splits_file, 'r'))
                for name, module in model.model.named_modules():
                    if isinstance(module, LinearWithMETDDYNAMICNLoRA) or isinstance(module, LinearWithMETDCOMPRESSLoRA) or isinstance(module, LinearWithMETDSIMILARITYLoRA):
                        rank = rank_config[name]
                        if training_config.dynamic_n_allocation:
                            n_split = n_splits_config[name]
                        else:
                            n_split = training_config.me_lora_n_split
                        module.init_method = 'vanilla'
                        module.dynamic_init(avg_rank=training_config.lora_rank, rank=rank, n_split=n_split)
            if (hasattr(training_config, 'use_lora_ga_pro') and training_config.use_lora_ga_pro):
                rank_config_file = os.path.join(os.path.dirname(args.ckpt), 'rank.json')
                rank_config = json.load(open(rank_config_file, 'r'))
                for name, module in model.model.named_modules():
                    if isinstance(module, LinearWithLoRAGAPro):
                        rank = rank_config[name]
                        module.prepare_init(allocated_rank=rank)
                        module.init_lora_weights()
        if args.ckpt is not None:
            load_ckpt(model=model.model, partial_ckpt_path=args.ckpt)
            print(f"loaded weight at{args.ckpt}")
        model = model.to(device).eval()
    print("Model loading done")
    print('======================================')
    print(f'The device of the model is {device}')
    print(f'The dtype of the model is {model_config.dtype}')
    print('======================================')
    # Print the prompts and results.

    if training_config:
        args.meta_prompt = training_config.meta_prompt
        args.prefix = training_config.prefix
        args.postfix = training_config.postfix
    print(f'Using meta prompt {args.meta_prompt}, prefix {args.prefix}, postfix {args.postfix}')

    if args.prompt is not None:
        result = model.generate(args.prompt, device, output_len=512)
        print('======================================')
        print(f'PROMPT: {args.prompt}')
        print(f'RESULT: {result}')
        print('======================================')

    if args.dataset_path is not None:
        if not args.result_path:
            args.result_path = floder_path
        format = args.dataset_path.split('.')[-1]
        read_func = format_dict[format]
        df = read_func(args.dataset_path)
        if args.read_num:
            df = stratified_sample(df, strata_col='task', total_sample_size=args.read_num)
        df['result'] = ''
        acc_count = 0
        w_count = 0
        disable = not args.tqdm
        iter_start = 0
        iter_end = df.shape[0]
        with tqdm(range(iter_start, iter_end, args.batch_size), desc='runing dataset', disable=disable) as tbar:
            for i in tbar:
                start, end = i, i+(args.batch_size-1)
                inputs = df.loc[start:end, 'input'].apply(lambda x: args.meta_prompt + args.prefix + x + args.postfix).to_list()
                # results = model.generate(inputs, device, output_len=args.output_len, eos=False, temperature=0)[0]
                results = model.generate(inputs, device, output_len=args.output_len, eos=False)[0]
                results = [result.strip("<|begin_of_text|>") for result in results]
                for idx, result in enumerate(results):
                    print(f"Index: {start+idx}, Result: {result}")  # Debugging each result
                    if result == df.loc[i+idx, 'output']:
                        acc_count += 1
                    else:
                        w_count += 1
                    df.loc[start+idx, 'result'] = result
                postfix={"acc":f'{round(acc_count/(end+1),4) * 100}%', 'wrong':f'{round(w_count/(end+1),4) * 100}%'}   
                tbar.set_postfix(postfix)
                if disable:
                    print(df.loc[start:end, ['output','result']])
                    print(postfix)
                if i % 1000 == 0:
                    # Save the file every 1000 steps.
                    df.to_json(os.path.join(args.result_path, 'result.json'), orient='records', force_ascii=False, lines=True)
        df.to_json(os.path.join(args.result_path, 'result.json'), orient='records', force_ascii=False, lines=True)
                
    if args.run_loop:
        while True:
            user_input = str(input(INPUT_PROMPT))
            if user_input == 'quit()':
                break
            elif user_input == 'output_len()':
                args.output_len = int(input("===>Please enter the output len:"))
            elif user_input == 'meta_prompt()':
                args.meta_prompt == str(input("===>Please enter the meta prompt:"))
            elif user_input == 'prefix()':
                args.prefix == str(input("===>Please enter the prefix:"))
            elif user_input == 'clear()':
                args.prefix = ''
                args.meta_prompt = ''
            else:
                # Make sure the meta prompt is correct.
                prompt = args.meta_prompt + args.prefix + user_input +args.postfix
                result = model.generate(prompt, device, output_len=args.output_len, eos=False, temperature=0)
                print('======================================')
                print(f'RESULT: {result}')
                print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--ckpt", type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/gue_except_covid_dp/final.ckpt')
    parser.add_argument("--ckpt", type=str, default=None)
    # parser.add_argument("--pretrained_ckpt", type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/dnallama_bpe_1em4/step_15000.ckpt')
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    # parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default='/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model')
    parser.add_argument("--meta_prompt", type=str, default='')
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--read_num", type=int, default=None)
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument("--prefix", type=str, default='<|start_header_id|>user<|end_header_id|>\n\n')
    parser.add_argument("--postfix", type=str, default='<|start_header_id|>assistant<|end_header_id|>\n\n')
    parser.add_argument("--model_name",
                        type=str,
                        default="llama3")
    parser.add_argument("--variant",
                        type=str,
                        default="8b",
                        choices=["test","2b", "7b", "8b"])
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--run_loop", action='store_true')
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument('--use_mos_lora', action='store_true')
    parser.add_argument("--use_dora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--me-lora-forward-method', type=str, default='for', choices=['for','einsum'])

    args = parser.parse_args()

    main(args)