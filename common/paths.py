"""please insert all paths into the paths dict"""
import os
import json
from common.registry import registry

mt_dir = os.path.split(os.getcwd())[0]
paths_file_path = os.path.join(mt_dir, "paths.json")
if not os.path.isfile(paths_file_path):
    raise FileNotFoundError("Please config paths in MyTransformers/paths.json!")
else:
    paths = json.load(open(paths_file_path, "r"))
    
for key, value in paths.items():
    for sub_k, sub_v in value.items():
        name = '_'.join([key, sub_k])
        registry.register_path(name=name, path=sub_v)