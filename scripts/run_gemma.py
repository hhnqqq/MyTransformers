import argparse
import contextlib
import random
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import torch

from tqdm import tqdm
from model import *
from common.registry import registry
from common.lora import switch_to_lora


@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def main(args):
    # Construct the model config.
    model_config = registry.get_model_config_class('_'.join([args.model_name, args.variant]))()
    model = registry.get_model_class(args.model_name)
    model_config.dtype = "float32" if args.device == "cpu" else "float16"
    model_config.quant = args.quant

    # Seed random.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create the model and load the weights.
    device = torch.device(args.device)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = model(model_config)
        if args.use_lora:
            switch_to_lora(model, replace_names=model_config.lora_layers, rank=args.lora_rank, use_dora=args.use_dora)
        model.load_weights(args.ckpt)
        model = model.to(device).eval()
    print("Model loading done")
    print('======================================')
    print(f'The device of the model is {device}')
    print(f'The dtype of the model is {model_config.dtype}')
    print('======================================')
    # Print the prompts and results.

    if args.prompt is not None:
        result = model.generate(args.prompt, device, output_len=4096)
        print('======================================')
        print(f'PROMPT: {args.prompt}')
        print(f'RESULT: {result}')
        print('======================================')

    if args.run_loop:
        while True:
            prompt = str(input('===>Please enter your prompt, or enter quit() to quit: '))
            if prompt == 'quit()':
                break
            else:
                result = model.generate(prompt, device, output_len=4096)
                print('======================================')
                print(f'PROMPT: {prompt}')
                print(f'RESULT: {result}')
                print('======================================')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default='/home/modelfun/zhaokangkang/mini_LLama/dnallama/output/pytorch_model.bin')
    parser.add_argument("--model-name",
                        type=str,
                        default="gemma")
    parser.add_argument("--variant",
                        type=str,
                        default="2b",
                        choices=["2b", "7b"])
    parser.add_argument("--device",
                        type=str,
                        default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument("--run_loop", action='store_true')
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--use_dora", action='store_true')
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--output_len", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--quant", action='store_true')
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    main(args)