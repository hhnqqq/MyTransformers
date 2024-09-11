import os
import json
import math
import torch
import logging
import deepspeed.comm as comm

from typing import Callable
from argparse import Namespace
from torch.utils.data import DataLoader

from common.utils import parallel_states as parallel_states
from common.utils import Timer, print_rank_0, ensure_directory_exists

class Trainer:
    def __init__(self, args, writer=None):
        self.args = args
        self.end = False
        self.writer = writer
        self.all_loss = 0.0
        self.global_step = 0
        self.all_metric = []
        self.eval_loss = None
        self.eval_metric = []
        self.best_eval_index = 0.0
        self.wait = 0
        self.save_floder = os.path.join(args.output_path, args.experiment_name)
        self.save_config = True
        ensure_directory_exists(self.save_floder, self.args.global_rank)

    def train(self, 
              model:torch.nn.Module, 
              train_data_loader:DataLoader, 
              forward_step: Callable, 
              optimizer: Callable = None,
              lr_scheduler: Callable = None,
              eval_data_loader:DataLoader = None,
              backward_step: Callable = None,
              eval_step: Callable = None,
              profiler: Callable = None,
              log_loss: bool = False):
        """
        Training loop for the model.

        Args:
            model (torch.nn.Module): Model to be trained.
            train_data_loader (DataLoader): Training data loader.
            eval_data_loader (DataLoader): Evaluation data loader.
            optimizer (Callable): Function for optimizer.
            forward_step (Callable): Forward step function.
            backward_step (Callable): Backward step function.
            eval_step: (Callable): Evaluation step function.
            log_loss (bool, optional): Flag to log loss values. Defaults to False.
        """
        print_rank_0('--->loaded the model, start training', self.args.global_rank)
        total_print = self.args.num_update_steps // self.args.show_avg_loss_step
        assert self.args.save_interval or self.args.save_epoch
        if self.args.save_epoch:
            update_steps_one_epoch = math.ceil(len(train_data_loader) / parallel_states.get_data_parallel_group())
            self.args.save_interval = (update_steps_one_epoch / (self.args.gradient_accumulation_steps)) * self.args.save_epoch
        with Timer(iterations=total_print) as timer:
            for step in range(self.args.num_update_steps):
                timer.average_time(entry='start')
                # Execute the forward step of the model and calculate the loss and metric.
                loss, metric = forward_step(model, train_data_loader, self.args, step)   
                # Accumulate loss and metric for gradient accumulation.
                if loss.isnan() or loss.isinf():
                    print(f'Skipping backward and optimizer step for nan or inf in forwarding loss at rank {self.args.global_rank}!')
                    self.all_loss += 0.0
                    self.all_metric.append({})
                else:
                    if 'loss_reduced' in metric.keys():
                        # If we have reduced loss, then report it rather than loss at rank 0.
                        self.all_loss += metric['loss_reduced'].item()
                    else:
                        self.all_loss += loss.item()
                    self.all_metric.append(metric)            
                    # Execute the backward step if provided (optional) to update model parameters.
                    if backward_step:
                        backward_step(model, optimizer, loss)     
                    if profiler:
                        profiler.step()     
                # Manage and print training information. 
                # Evaluate the model at intervals specified by eval_interval.
                if (step + 1) % self.args.eval_interval == 0 and eval_step is not None:
                    assert eval_data_loader is not None, 'evaluation dataset can not be None'
                    self.eval_loss, eval_metric = eval_step(model, eval_data_loader, self.args, step)
                    self.eval_metric.append(eval_metric)
                self.info_manager(step, timer, log_loss)
                self.save_model(model, optimizer, lr_scheduler, step)  
                if self.end:
                    print_rank_0("Early stopping triggered.", self.args.global_rank)
                    break
        # Mark the end of training loop to control the save behavior in save_model function.
        self.end = True
        self.save_model(model, optimizer, lr_scheduler, step)
        print_rank_0(f"--->Total time consumed: {timer.time_cost}", self.args.global_rank)

    def earily_stop(self):
        index = self.args.earily_stop_index
        if index == 'loss':
            if self.best_eval_index is None:
                self.best_eval_index = self.eval_loss
            elif self.best_eval_index > self.eval_loss:
                self.best_eval_index = self.eval_loss
                self.wait = 0
            else:
                self.wait += 1
        else:
            eval_metric = self.eval_metric[0]
            if index in eval_metric.keys():
                if self.best_eval_index is None:
                    self.best_eval_index = eval_metric[index]
                elif eval_metric[index] > self.best_eval_index:
                    self.best_eval_index = eval_metric[index]
                else:
                    self.wait += 1
        if self.wait > self.args.earily_stop_patience:
            self.end = True

    def info_manager(self, step:int, timer:Timer, log_loss: bool = False):
        # Determine the level of logging based on log_loss flag.
        loss_level = logging.INFO if log_loss else logging.DEBUG

        if self.args.local_rank == 0:
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                self.global_step += 1
                if (self.global_step) % self.args.show_avg_loss_step == 0:
                    timer.average_time(entry='end')
                    # Calculate average time per step and average loss.
                    avg_time = (timer.loop_time) / self.args.show_avg_loss_step
                    avg_loss = self.all_loss / self.args.show_avg_loss_step
                    remaining_time = timer.calculate_remaining_time()
                    # Prepare the string to print with step number, average loss, and time.
                    print_str = f"--->step={self.global_step}, avg_loss={avg_loss:.8f}, avg_time={avg_time:.2f}s"
                    print_time_str = f", remaining_time={remaining_time}, remaining_steps:{self.args.num_update_steps - step}"

                    # Log the average loss to Tensorboard.
                    if self.writer is not None and self.args.global_rank == 0:
                        self.writer.add_scalar('loss', avg_loss, self.global_step)

                    # Include additional information specific to the task if available.
                    if self.get_task_print:
                        print_str += self.get_task_print(self.all_metric, self.args) + print_time_str
                    else:
                        print_str += print_time_str
                    # Print the information with the determined logging level.
                    print_rank_0(print_str, self.args.global_rank, loss_level)
                    self.all_loss = 0.0
                    self.all_metric = []

            if (step + 1) % self.args.eval_interval == 0 and self.eval_loss:
                if self.eval_loss is None:
                    print_str = "Eval loss is None!"
                else:
                    print_str = f"--->step={self.global_step}, eval_loss={self.eval_loss:.4f}"
                if self.get_task_print and self.eval_loss is not None:
                    print_str += self.get_task_print(self.eval_metric, self.args)   
                print_rank_0(print_str, self.args.global_rank, loss_level)
                if self.writer is not None and self.args.global_rank == 0:
                    self.writer.add_scalar('eval_loss', self.eval_loss, self.global_step)
                self.eval_loss = 0.0
                self.eval_metric = []

    def register_task_print(self, print_func):
        self.task_print = print_func

    @property
    def get_task_print(self):
        return getattr(self, "task_print", None)

    def save_model(self, model, optimizer, lr_scheduler, step): 
        config_path = os.path.join(self.save_floder, 'config.json')
        if self.save_config and isinstance(self.args, Namespace):
            with open(config_path, 'w', encoding='utf-8') as f:
                print_rank_0(f'--->Saving training config at {step+1}th step in {config_path}.', self.args.global_rank)
                save_dict = {k:v for k,v in self.args.__dict__.items() if k != 'device'}
                json.dump(save_dict, f)
                self.save_config = False


        if self.args.num_pp_stages is not None:
            if not self.end and (step+1) % self.args.save_interval == 0: 
                print_rank_0(f'--->Start saving model at {step+1}th step in {self.save_floder}.', self.args.global_rank)
                model.save_checkpoint(self.save_floder, tag=f'step_{step+1}')
                print_rank_0('--->Saved the model.', self.args.global_rank)
            elif self.end and (step+1) % self.args.save_interval != 0:
                print_rank_0(f'--->Start saving model at final step in {self.save_floder}.', self.args.global_rank)
                model.save_checkpoint(self.save_floder, tag='final')
                print_rank_0('--->Saved the model.', self.args.global_rank)
            return

        if self.args.global_rank <= 0: 
            if not self.end and (step+1) % self.args.save_interval == 0: 
                save_path = os.path.join(self.save_floder, f'step_{step+1}.ckpt')
                print_rank_0(f'--->Start saving model at {step+1}th step in {save_path}.', self.args.global_rank)
                self.torch_save(model, optimizer, lr_scheduler, save_path)
                print_rank_0('--->Saved the model.', self.args.global_rank)  
            elif self.end: 
                save_path = os.path.join(self.save_floder, 'final.ckpt')
                print_rank_0(f'--->Start saving model at final step in {save_path}.', self.args.global_rank)
                self.torch_save(model, optimizer, lr_scheduler, save_path)
                print_rank_0('--->Saved the model.', self.args.global_rank)  

    def torch_save(self, model:torch.nn.Module, optimizer, lr_scheduler, save_path):
        if self.args.save_trainable:
            trainable_params = {}
            for name, param in model.module.named_parameters():
                if param.requires_grad:
                    trainable_params[name] = param.data
            model_state_dict = trainable_params
        else:
            model_state_dict = model.module.state_dict()
        
        if optimizer and lr_scheduler:
            ckpt_to_save = {'model_state_dict':model_state_dict,
                            'optimizer_state_dict':optimizer.state_dict(),
                            'lr_scheduler_state_dict':lr_scheduler.state_dict()}
        else:
            ckpt_to_save = model_state_dict
        torch.save(ckpt_to_save, save_path)
    
if __name__ == '__main__':
    """A quick test for trainer"""
    import os
    import torchvision
    from torchvision import datasets, transforms
    from dataclasses import dataclass
    import traceback

    os.environ['NO_LOG_FILE'] = 'true'
    model = torchvision.models.resnet18()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fake_dataset = datasets.FakeData(size=100, num_classes=3, transform=transforms.ToTensor())
    test_dataset = datasets.FakeData(size=200, num_classes=3, transform=transforms.ToTensor())
    data_loader = iter(DataLoader(fake_dataset, batch_size=10, shuffle=True))
    test_data_loader = iter(DataLoader(test_dataset, batch_size=10, shuffle=True))

    # Define forward and backward step functions
    def forward_step(model, data_loader, args, step):
        inputs, labels = next(data_loader)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        metric = {"acc": accuracy(outputs, labels)}
        return loss, metric

    def eval_step(model, data_loader, args, step):
        inputs, labels = next(data_loader)
        outputs = model(inputs)
        loss = torch.nn.CrossEntropyLoss()(outputs, labels)
        return loss.item()


    def backward_step(_, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def accuracy(outputs, labels):
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum().item()
        total = labels.size(0)
        return correct / total
    
    def task_print(all_metric, args):
        acc_count = sum([sub_dict['acc'] for sub_dict in all_metric])
        return f' train_acc:{(acc_count/args.show_loss_step) * 100}%'

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    @dataclass
    class ARGS:
        num_update_steps = 10000
        show_loss_step = 10
        save_interval = 10000
        eval_interval = 100000
        output_path = '.'
        experiment_name = 'resnet'
        global_rank = 0
        local_rank = 0
        gradient_accumulation_steps = 1
        
    trainer = Trainer(args=ARGS, writer=None)
    trainer.register_task_print(task_print)
    try:
        trainer.train(model=model, 
                      train_data_loader=data_loader, 
                      eval_data_loader=test_data_loader,
                      optimizer=optimizer, 
                      forward_step=forward_step, 
                      eval_step=eval_step,
                      backward_step=backward_step)
    except:
        traceback_info = traceback.format_exc()
        print_rank_0(traceback_info, ARGS.global_rank ,level=logging.ERROR)
