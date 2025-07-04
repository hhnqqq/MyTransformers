import os
import json
import argparse

import torch
import pandas as pd

from functools import partial

from model import *
from common.registry import registry
from common.lora_modules import *
from common.utils import set_random_seed, load_ckpt
from common.utils.utils import set_default_tensor_type
from common.utils import parallel_states as parallel_states

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
Enter clear() to clear meta prompt and prefix: \n'''

def save_results(df, result_path):
    os.makedirs(result_path, exist_ok=True)
    df.to_json(
        os.path.join(result_path, 'result.json'),
        orient='records',
        force_ascii=False,
        lines=True
    )

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

    training_args = None
    if os.path.exists(floder_path) and os.path.isdir(floder_path):
        config_path = os.path.join(floder_path, 'config.json')
        if os.path.exists(config_path):
            training_args = Namespace(**json.load(open(config_path, 'r')))
            
    model_config = registry.get_model_config_class('_'.join([args.model_name, args.variant]))()
    model: nn.Module = registry.get_model_class(args.model_name)
    if training_args:
        if training_args.bf16:
            model_config.dtype = "bf16" 
        elif training_args.fp16:
            model_config.dtype = "fp16"
    else:
        model_config.dtype = "fp32"
    model_config.quant = args.quant
    model_config.tokenizer = args.tokenizer if args.tokenizer is not None else training_args.tokenizer_path 
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
            print(f"loaded pretrained weight at {args.pretrained_ckpt}")
        if training_args and training_args.use_lora:
            print("Replacing model with lora layers")
            prepare_lora_for_inference(model, training_args)
        if args.ckpt is not None:
            load_ckpt(model=model.model, partial_ckpt_path=args.ckpt)
            print(f"loaded weight at{args.ckpt}")
            if check_shared_lora_weights_required:
                create_shared_weight_references(model)
        model = model.to(device).eval()
    print("Model loading done")
    print('======================================')
    print(f'The device of the model is {device}')
    print(f'The dtype of the model is {model_config.dtype}')
    print('======================================')
    # Print the prompts and results.

    if training_args:
        args.meta_prompt = training_args.meta_prompt
        args.prefix = training_args.prefix
        args.postfix = training_args.postfix
    print(f'Using meta prompt {args.meta_prompt}, prefix {args.prefix}, postfix {args.postfix}')

    if args.prompt is not None:
        result = model.generate(args.prompt, device, output_len=512)
        print('======================================')
        print(f'PROMPT: {args.prompt}')
        print(f'RESULT: {result}')
        print('======================================')

    if args.dataset_path is not None:   

        # Initialize result path
        args.result_path = args.result_path or floder_path
        
        # Read and prepare dataset
        file_format = args.dataset_path.split('.')[-1]
        df = format_dict[file_format](args.dataset_path)
        
        if args.read_num:
            df = stratified_sample(df, strata_col='task', total_sample_size=args.read_num)
        
        df['result'] = ''
        acc_count = w_count = 0
        disable_tqdm = not args.tqdm
        total_samples = df.shape[0]
        
        # Process in batches
        with tqdm(range(0, total_samples, args.batch_size), 
                desc='Processing dataset', 
                disable=disable_tqdm) as tbar:
            
            for batch_start in tbar:
                batch_end = min(batch_start + args.batch_size - 1, total_samples - 1)
                batch_indices = range(batch_start, batch_end + 1)
                
                # Prepare inputs
                inputs = [
                    args.meta_prompt + args.prefix + x + args.postfix
                    for x in df.loc[batch_indices, 'input']
                ]
                
                # Generate results
                results = model.generate(
                    inputs, 
                    device, 
                    output_len=args.output_len, 
                    eos=False, 
                    temperature=args.temperature,
                    top_p=args.top_p
                )[0]
                
                # Process results
                results = [r.strip("<|begin_of_text|>") for r in results]
                
                for idx, (result, true_output) in enumerate(zip(
                    results, 
                    df.loc[batch_indices, 'output']
                )):
                    current_idx = batch_start + idx
                    if result == true_output:
                        acc_count += 1
                    else:
                        w_count += 1
                    
                    df.loc[current_idx, 'result'] = result
                    
                    if disable_tqdm:
                        print(f"Index: {current_idx}, Result: {result}")
                
                # Update progress
                accuracy = round(acc_count/(batch_end + 1), 4) * 100
                wrong_rate = round(w_count/(batch_end + 1), 4) * 100
                tbar.set_postfix({
                    "accuracy": f"{accuracy}%", 
                    "wrong": f"{wrong_rate}%"
                })
                
                if disable_tqdm:
                    print(df.loc[batch_indices, ['output', 'result']])
                    print(f"Accuracy: {accuracy}%, Wrong: {wrong_rate}%")
                
                # Periodic save
                if batch_start % 1000 == 0:
                    save_results(df, args.result_path)
        
        # Final save
        save_results(df, args.result_path)
                
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
                result = model.generate(prompt, 
                                        device, 
                                        output_len=args.output_len, 
                                        eos=False, 
                                        temperature=args.temperature,
                                        top_p=args.top_p)[0][0]
                print('======================================')
                print(f'RESULT: {result}')
                print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=None, required=True)
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
    parser.add_argument("--output_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)

    args = parser.parse_args()

    main(args)