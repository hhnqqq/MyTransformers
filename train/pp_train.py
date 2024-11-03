from argparse import Namespace
from torch.profiler import record_function
from deepspeed.runtime.pipe.engine import PipelineEngine

from dataset_classes import RepeatingLoader

def forward_step_pipeline(model: PipelineEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with record_function("forward_backward_path"):
        return model.train_batch(data_loader), []

def eval_step_pipeline(model: PipelineEngine, data_loader: RepeatingLoader, args: Namespace, step: int):
    with record_function("eval_path"):
        return model.eval_batch(data_loader).item()
    
def task_print_pipeline(all_metric, args):
    return ''