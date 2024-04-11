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
              forward_step, 
              backward_step, 
              args):
        print_rank_0('--->loaded the model, start training', args.global_rank)
        with Timer() as timer:
            for step in range(args.num_update_steps):
                timer.average_time(entry='start')
                loss, metric = forward_step(model, data_loader, self.args)
                if backward_step:
                    backward_step(model, loss)
                self.all_loss += loss.item()
                self.all_metric.append(metric)
                self.info_manager(step, timer, args)
                self.save_model(model, step, args)
        self.end = True
        self.save_model(model, step, args)
        print_rank_0(f"--->total time cosumed is {timer.time_cost}", args.global_rank)

    def info_manager(self, step:int, timer:Timer, args):
        if args.local_rank == 0:
            if (step+1) % args.gradient_accumulation_steps == 0:
                self.global_step += 1
                if (self.global_step + 1) % args.show_loss_step == 0:
                    timer.average_time(entry='end')
                    avg_time = (timer.loop_time) / args.show_loss_step
                    avg_loss = self.all_loss / args.show_loss_step
                    if self.writer is not None:
                        self.writer.add_scalar('loss', avg_loss, self.global_step)
                    print_str = f"--->step={self.global_step+1}, avg_loss={avg_loss:.4f}, avg_time={avg_time:.2f}s"
                    if self.get_task_print:
                        print_str += self.get_task_print(self, step, args)
                    print_rank_0(print_str)
                    self.all_loss = 0.0

    def register_task_print(self, print_func):
        self.task_print = print_func

    @property
    def get_task_print(self):
        return getattr(self, "task_print", None)

    def save_model(self, model, step, args): 
        if args.global_rank <= 0: 
            if not self.end and (step+1) % args.save_interval == 0: 
                if step % args.save_interval == 0:
                    torch.save(model.state_dict(), os.path.join(args.output_path, f'{args.experiment_name}_{step}.ckpt'))
            else: 
                torch.save(model.state_dict(), os.path.join(args.output_path, f'{args.experiment_name}_final.ckpt'))
            
    
if __name__ == '__main__':
    pass