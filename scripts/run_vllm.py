import os
import ray
import gc
import json
import vllm
import math
import asyncio
import argparse
import warnings
import pandas as pd

from jinja2 import Template
from functools import partial
from argparse import Namespace
from vllm.utils import get_open_port
from vllm.lora.request import LoRARequest
from vllm.sampling_params import SamplingParams
from vllm import AsyncLLMEngine, AsyncEngineArgs
from transformers import AutoTokenizer, AutoModelForCausalLM

from common.utils.utils import load_ckpt
from common.lora_modules import switch_to_lora, LinearWithLoRA, LinearWithGoRA

SYSTEM_PROMPTS = {"qwq":"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. """,
"gora":"A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>."}

INSTRUCTION_TEMPLATES = {"qwq":"""Please reason step by step, and put your final answer within \\boxed{} for closed-end problem (including multiple choice problem).
If the problem requires multiple answers to be answered, place all final answers in one \\boxed{} environment, separated by commas.
If the problem is in Chinese, provide your reasoning and answer in Chinese. Otherwise, use English.
Problem: {{prompt}}""",
"gora":"Please reason step by step, and put your final answer within \\boxed{} for closed-end problem (including multiple choice problem). This is the problem: {{prompt}}"}

DATA_PARALLEL_WORLD_SIZE = int(os.environ.get('DATA_PARALLEL_WORLD_SIZE', "1"))
TENSOR_PARALLEL_WORLD_SIZE = int(os.environ.get('TENSOR_PARALLEL_WORLD_SIZE', "1"))
WORLD_SIZE = DATA_PARALLEL_WORLD_SIZE * TENSOR_PARALLEL_WORLD_SIZE
if WORLD_SIZE > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(WORLD_SIZE))
        

read_mapping = {'csv':pd.read_csv,
                'json':pd.read_json,
                'xlsx': pd.read_excel,
                'parquet': pd.read_parquet,
                'jsonl':partial(pd.read_json, lines=True)}

def get_sd_from_mt_lora(model_path, lora_path):
    model = AutoModelForCausalLM.from_pretrained(model_path)
    config = json.load(open(os.path.join(lora_path, 'config.json'), 'r'))
    config = Namespace(**config)
    switch_to_lora(model, config)

    if hasattr(config, 'use_gora') and config.use_gora:
        rank_config_file = os.path.join(lora_path, 'rank.json')
        rank_config = json.load(open(rank_config_file,'r'))
        for name,module in model.named_modules():
            if isinstance(module, LinearWithGoRA):
                rank = rank_config[name]
                module.init_method = 'vanilla'
                module.dynamic_init(config.lora_rank, rank)

    load_ckpt(model, partial_ckpt_path=os.path.join(lora_path, 'final.ckpt'), ignore_missing_keys=True)
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.merge_and_del()
            print(f'Merged LoRA weight for layer: {name}')
    model_sd = [(name, data) for name, data in model.state_dict().items()]
    del model
    gc.collect()
    return model_sd

def load_df(path):
    path_info = path.split('/')[-1]
    name, suffix = path_info.split('.')
    df = read_mapping.get(suffix)(path)
    df['dataset'] = name
    return df

def tokenize(prompts,
             tokenizer:AutoTokenizer,
             args,
             encode=True):
    if not isinstance(prompts, list):
        prompts = [prompts]

    tokenized_prompts = []
    system_prompt = SYSTEM_PROMPTS.get(args.template_type, "")
    instruction_template = INSTRUCTION_TEMPLATES.get(args.template_type, "")
    apply_instruction = args.apply_instruction and instruction_template
    apply_system_prompt = args.apply_system_prompt and system_prompt

    for prompt in prompts:
        if apply_instruction:
            prompt = Template(instruction_template).render(prompt=prompt)
        system_message = [{'content':system_prompt, 'role':'system'}]
        user_message = [{'content':prompt, 'role':'user'}]
        messages = system_message + user_message if apply_system_prompt else user_message
        try:
            tokenized_prompt = tokenizer.apply_chat_template(messages,
                                                            tokenize=False,
                                                            add_generation_prompt=True)
            if '<think>' not in tokenized_prompt:
                tokenized_prompt += '<think>'
            if encode:
                tokenized_prompt = tokenizer(tokenized_prompt).input_ids
            tokenized_prompts.append(tokenized_prompt)
        except:
            pass
    return tokenized_prompts
        
async def generate(model_name,
                   lora_request,
                   sampling_params,
                   args):
    if DATA_PARALLEL_WORLD_SIZE > 1:
        warnings.warn('Data Parallelism is not competible with streaming generation, ignoring.')

    if args.load_lora_from_mt:
        model_sd = get_sd_from_mt_lora(args.model_name, args.lora_path)

    engine_args = AsyncEngineArgs(model=model_name,
                                  tensor_parallel_size=TENSOR_PARALLEL_WORLD_SIZE,
                                  enable_chunked_prefill=True,
                                  enable_prefix_caching=True,
                                  max_model_len=args.max_model_len)
    engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(
        engine_args=engine_args
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.load_lora_from_mt:
        engine.model_executor.driver_worker.worker.model_runner.model.load_weights(model_sd)

    while True:
        prompt = input('请输入你的问题：')
        request_id = id(prompt)
        prompt = tokenize(prompt, tokenizer, args, encode=False)[0]
        print(f"用户: {prompt}")
        previous_token = ""
        print("助手：", end=" ", flush=True)

        async for output in engine.generate(prompt, 
                                            sampling_params, 
                                            request_id,
                                            lora_request=lora_request):
            token = output.outputs[0].text
            new_token = token[len(previous_token):]
            print(new_token, end="", flush=True)
            previous_token = token
        
        print()

@ray.remote(num_gpus=TENSOR_PARALLEL_WORLD_SIZE)
def inference(model_name,
              lora_request,
              prompts,
              sampling_params,
              dp_rank,
              args):
    os.environ["VLLM_DP_RANK"] = str(dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(DATA_PARALLEL_WORLD_SIZE)
    os.environ["VLLM_DP_MASTER_IP"] = args.dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(args.dp_master_port)
        
    if args.load_lora_from_mt:
        model_sd = get_sd_from_mt_lora(args.model_name, args.lora_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_prompts = tokenize(prompts, tokenizer, args)
    llm = vllm.LLM(model=model_name,
                   tensor_parallel_size=TENSOR_PARALLEL_WORLD_SIZE,
                   max_model_len=args.max_model_len)
    
    if args.load_lora_from_mt:
        llm.llm_engine.model_executor.driver_worker.worker.model_runner.model.load_weights(model_sd)

    outputs = llm.generate(prompt_token_ids=tokenized_prompts,
                           sampling_params=sampling_params,
                           use_tqdm=True if dp_rank == 0 else False,
                           lora_request=lora_request)
    outputs = [output.outputs[0].text for output in outputs]
    print(f'dp rank: {dp_rank} finishied generation.')
    return dict(zip(prompts, outputs))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name or path of the model')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path of LoRA weights')
    parser.add_argument('--chat', action='store_true',
                        help='Wether to chat or inference.')
    parser.add_argument('--top_p', type=float, default=0.1)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--max_tokens', type=int, default=16384*1.5)
    parser.add_argument('--max_model_len', type=int, default=16384*1.5)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--apply_instruction', action='store_true',
                        help='Wether to use instruction template.')
    parser.add_argument('--apply_system_prompt', action='store_true',
                        help='Wether to use system prompt.')
    parser.add_argument('--template_type', type=str, default='p1')
    parser.add_argument('--save_name', type=str, default='test_result.jsonl',
                        help='Name of file to save.')
    parser.add_argument('--read_num', type=int, default=None)
    parser.add_argument('--repetition_penalty', type=float, default=1.1)
    parser.add_argument('--load_lora_from_mt', action='store_true')
    parser.add_argument('--stop', type=str, default='</answer>')
    parser.add_argument('--datasets', action='append', default=None)
    args = parser.parse_args()
    
    sampling_params = SamplingParams(top_p=args.top_p,
                                     temperature=args.temperature,
                                     max_tokens=args.max_tokens,
                                     seed=42,
                                     stop=args.stop,
                                     include_stop_str_in_output=True,
                                     repetition_penalty=args.repetition_penalty)
    lora_request = LoRARequest(lora_name=args.lora_path) if (args.lora_path and not args.load_lora_from_mt) else None


    if args.chat:
        asyncio.run(generate(model_name=args.model_name,
                             lora_request=lora_request,
                             sampling_params=sampling_params,
                             args=args))
    else:
        dfs = [load_df(dataset) for dataset in args.datasets]
        df = pd.concat(dfs, axis=0)
        sample_kwargs = {'n':args.read_num} if args.read_num else {'frac':1}
        df = df.sample(**sample_kwargs)
        df.drop_duplicates(subset=['problem'], inplace=True)
        df.reset_index(inplace=True, drop=True)
        prompts = df.problem.to_list()
        num_prompts = len(prompts)

        args.dp_master_ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        args.dp_master_port = os.environ.get('MASTER_PORT', get_open_port())
        num_prompts_per_rank = math.ceil(len(prompts) / DATA_PARALLEL_WORLD_SIZE)
        ray_outputs = []
        for data_parallel_rank in range(DATA_PARALLEL_WORLD_SIZE):
            start = data_parallel_rank * num_prompts_per_rank
            end = min(num_prompts, (data_parallel_rank + 1) * num_prompts_per_rank)
            prompts_this_rank = prompts[start:end]
            ray_outputs.append(inference.remote(
                                args.model_name,
                                lora_request,
                                prompts_this_rank,
                                sampling_params,
                                data_parallel_rank,
                                args))

        output_qa_dict = {}
        for sub_dict in ray_outputs:
            output_qa_dict.update(ray.get(sub_dict))

        df['test_result'] = df.problem.map(output_qa_dict)
        save_path = args.save_path if args.save_path else os.path.dirname(args.model_name)
        os.makedirs(save_path, exist_ok=True)
        save_path = os.path.join(save_path, args.save_name)
        df.to_json(save_path, orient='records', lines=True)
        print(f'Saved generated data to {save_path}')
        
if __name__ == '__main__':
    # important
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    main()
    
