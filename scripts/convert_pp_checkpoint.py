import torch
from pathlib import Path
from os.path import join
import argparse

from model import *
from common.registry import registry


def convert_model_to_hf(args):
    model_static_dict = {}
    model_config = registry.get_model_config_class(name='-'.join([args.model_name, args.variant]))()
    n_layers = model_config.num_hidden_layers + 2
    for path in Path(args.pipeline_model_dir).iterdir():
        print("已经处理文件：{}".format(path))
        if not path.name.startswith('layer'):
            continue
        small_static_dict = torch.load(path, map_location="cpu")
        layer_i = int(path.name.split('-')[0].replace('layer_', ''))
        if layer_i == 0:
            model_static_dict["embedder.weight"] = small_static_dict["embedder.weight"]
        elif layer_i == n_layers -1 :
            for k, v in small_static_dict.items():
                model_static_dict["model.norm.weight"] = v
        else:
            for k, v in small_static_dict.items():
                model_static_dict["model." + k.replace("layer.", "layers.{}.".format(layer_i - 1))] = v

    torch.save(model_static_dict, join(args.save_model_dir, "pytorch_model.bin"))

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipeline_model_dir', default='/home/modelfun/zhaokangkang/mini_LLama/dnallama/output/dna_test_20', type=str, help='')
    parser.add_argument('--save_model_dir', default='/home/modelfun/zhaokangkang/mini_LLama/dnallama/output/', type=str, help='')
    parser.add_argument('--model_name', default='gemma', type=str, help='')
    parser.add_argument('--variant', default='2b', type=str, help='')
    return parser.parse_args()


args = set_args()
convert_model_to_hf(args.pipeline_model_dir, args.save_model_dir)