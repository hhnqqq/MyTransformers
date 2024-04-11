import os
import torch
from torch.utils.data import DataLoader
from gemma.utils import Timer, print_rank_0

class Trainer:
    def __init__(self, args, writer=None):
        self.args = args
        self.end = False
        self.writer = writer
        self.all_loss = 0.0
        self.global_step = 0
        self.all_metric = []

    def train(self, 
              model:torch.nn.Module, 
              data_loader:DataLoader, 
              optimizer,
              forward_step, 
              backward_step):
        print_rank_0('--->loaded the model, start training', self.args.global_rank)
        with Timer() as timer:
            for step in range(self.args.num_update_steps):
                timer.average_time(entry='start')
                loss, metric = forward_step(model, data_loader, self.args)
                if backward_step:
                    backward_step(model, optimizer, loss)
                self.all_loss += loss.item()
                self.all_metric.append(metric)
                self.info_manager(step, timer)
                self.save_model(model, step)
        self.end = True
        self.save_model(model, step)
        print_rank_0(f"--->total time cosumed is {timer.time_cost}", self.args.global_rank)

    def info_manager(self, step:int, timer:Timer):
        if self.args.local_rank == 0:
            if (step+1) % self.args.gradient_accumulation_steps == 0:
                self.global_step += 1
                if (self.global_step + 1) % self.args.show_loss_step == 0:
                    timer.average_time(entry='end')
                    avg_time = (timer.loop_time) / self.args.show_loss_step
                    avg_loss = self.all_loss / self.args.show_loss_step
                    if self.writer is not None:
                        self.writer.add_scalar('loss', avg_loss, self.global_step)
                    print_str = f"--->step={self.global_step+1}, avg_loss={avg_loss:.4f}, avg_time={avg_time:.2f}s"
                    if self.get_task_print:
                        print_str += self.get_task_print(self.all_metric, self.args)
                    print_rank_0(print_str)
                    self.all_loss = 0.0
                    self.all_metric = []

    def register_task_print(self, print_func):
        self.task_print = print_func

    @property
    def get_task_print(self):
        return getattr(self, "task_print", None)

    def save_model(self, model, step): 
        if self.args.global_rank <= 0: 
            if not self.end and (step+1) % self.args.save_interval == 0: 
                save_path = os.path.join(self.args.output_path, f'{self.args.experiment_name}_{step+1}.ckpt')
                print_rank_0(f'--->Start saving model at {step+1}th step in {save_path}.')
                torch.save(model.state_dict(), save_path)
                print_rank_0('--->Saved the model.')
            elif self.end: 
                save_path = os.path.join(self.args.output_path, f'{self.args.experiment_name}_final.ckpt')
                print_rank_0(f'--->Start saving model at final step in {save_path}.')
                torch.save(model.state_dict(), save_path)
            
    
if __name__ == '__main__':
    """A quick test for trainer"""
    import torchvision
    from torchvision import datasets, transforms
    from dataclasses import dataclass

    model = torchvision.models.resnet18()

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    fake_dataset = datasets.FakeData(size=100, num_classes=3, transform=transforms.ToTensor())
    data_loader = DataLoader(fake_dataset, batch_size=10, shuffle=True)
    
    # Define forward and backward step functions
    def forward_step(model, data_loader, _):
        for inputs, labels in data_loader:
            outputs = model(inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            metric = {"acc": accuracy(outputs, labels)}
            return loss, metric

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
        output_path = '.'
        experiment_name = 'resnet'
        global_rank = 0
        local_rank = 0
        gradient_accumulation_steps = 1
        
    trainer = Trainer(args=ARGS, writer=None)
    trainer.register_task_print(task_print)
    trainer.train(model, data_loader, optimizer, forward_step, backward_step)
