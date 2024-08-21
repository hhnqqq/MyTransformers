import json
import torch
import shutil
import argparse
from pathlib import Path
from os.path import join

from model import *
from common.registry import registry
from common.utils import get_merged_state_dict
from common.lora_modules import switch_to_lora, LinearWithLoRA


def convert_model_to_hf(args):
    model_config = registry.get_model_config_class(name='_'.join([args.model_name, args.variant]))()
    model_config.tokenizer = args.tokenizer
    # Number of layers of model, plus 2 additional pipeline layers
    n_layers = model_config.num_hidden_layers + 2
    if args.save_name is None:
        args.save_name = args.model_path.split('/')[-1] + '.ckpt'
    if '.ckpt' not in args.save_name:
        args.save_name += '.ckpt'
    parent_dir = os.path.dirname(args.model_path)
    train_config_path = os.path.join(parent_dir, 'config.json')
    if os.path.exists(train_config_path):
        train_config = json.load(open(train_config_path, encoding='utf-8'))
        train_config = argparse.Namespace(**train_config)
    else:
        train_config = None

    if args.pipeline_model:
        # Pipeline model checkpoint should be merged.
        model_state_dict = {}
        for path in Path(args.model_path).iterdir():
            print("已经处理文件：{}".format(path))
            if not path.name.startswith('layer'):
                continue
            small_state_dict = torch.load(path, map_location="cpu")
            layer_i = int(path.name.split('-')[0].replace('layer_', ''))
            for k,v in small_state_dict.items():
                if args.model_name == 'gemma':
                    if layer_i == 0:
                        model_state_dict["embedder.weight"] = small_state_dict["embedder.weight"]
                    elif layer_i == n_layers -1 :
                        for k, v in small_state_dict.items():
                            model_state_dict["model.norm.weight"] = v
                    else:
                        for k, v in small_state_dict.items():
                            model_state_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1))] = v
                elif 'llama' in args.model_name:
                    if layer_i == 0:
                        model_state_dict["tok_embeddings.weight"] = small_state_dict["embedder.weight"]
                    elif layer_i == n_layers -1 :
                        model_state_dict["norm.weight"] = small_state_dict["final_norm.weight"]
                        model_state_dict["output.weight"] = small_state_dict["o_proj.weight"]
                    else:
                        for k, v in small_state_dict.items():
                            model_state_dict[k.replace("layer.", "layers.{}.".format(layer_i - 1))] = v

    if args.pretrained_model_path:
        # Merge trainable param with freeze param.
        model_state_dict,_,_ = get_merged_state_dict(args.pretrained_model_path, args.model_path)

    if not args.not_merge_lora and train_config is not None and (train_config.use_lora or train_config.use_lora_plus or train_config.use_dora):

        print('合并lora权重中')
        print('开始加载模型')
        model_cls = registry.get_model_class(args.model_name)
        model = model_cls(model_config)
        print('替换lora层')
        switch_to_lora(model,
                    replace_names=train_config.replace_modules,
                    rank=train_config.lora_rank,
                    use_dora=train_config.use_dora,
                    plora_steps=None)
        model.model.load_state_dict(model_state_dict, strict=False)
        print('合并lora权重')
        for module in model.modules():
            if isinstance(module, LinearWithLoRA):
                module.merge_and_del()
        print('转为指定dtype')
        if args.fp16:
            model.float16()
        else:
            model.bfloat16()
        torch.save(model.model.state_dict(), join(args.save_model_dir, args.save_name))
    else:
        torch.save(model_state_dict, join(args.save_model_dir, args.save_name))
    if args.rm_origin:
        shutil.rmtree(args.model_dir)


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_path',default='/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2.pth', type=str, help='')
    parser.add_argument('--model_path',default='/home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/llama2_gue_pretrain_2k/final.ckpt', type=str, help='')
    parser.add_argument('--pipeline_model', action='store_true')
    parser.add_argument('--save_model_dir', default='/home/bingxing2/ailab/scx6mh7/workspace/dnallama/output/llama2_gue_pretrain_2k/', type=str, help='')
    parser.add_argument('--tokenizer', default='/home/bingxing2/ailab/scx6mh7/workspace/llama/llama2_tokenizer.model',type=str)
    parser.add_argument('--save_name', default='final_full', type=str, help='')
    parser.add_argument('--model_name', default='llama2', type=str, help='')
    parser.add_argument('--variant', default='7b', type=str, help='')
    parser.add_argument('--rm_origin', action='store_true')
    parser.add_argument('--not_merge_lora', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    return parser.parse_args()


args = set_args()
convert_model_to_hf(args)