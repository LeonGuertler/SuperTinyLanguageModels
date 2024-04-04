"""A profiler that trains on a subset of the data and keeps
track of relevant statistics (like memory and time usage of 
each module)."""

import time

import torch
import wandb
from omegaconf import OmegaConf
from trainers import utils
from trainers.base_trainer import BaseTrainer

import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function

class TimeWrapper:
    def __init__(self, func):
        self.func = func
        self.time_tracker = []
    
    def __call__(self, *args, **kwargs):
        t0 = time.time()
        out = self.func(*args, **kwargs)
        t1 = time.time()
        self.time_tracker.append(t1 - t0)
        return out
    
    def get_time(self):
        return self.time_tracker
    

class TimeWrapperModel:
    def __init__(self, model):
        self.model = model
        self.time_tracker_train = []
        self.time_tracker_eval = []


    def __call__(self, *args, **kwargs):
        t0 = time.time()
        out = self.model(*args, **kwargs)
        t1 = time.time()
        if self.model.training:
            self.time_tracker_train.append(t1 - t0)
        else:
            self.time_tracker_eval.append(t1 - t0)
        return out
    
    def get_time_train(self):
        return self.time_tracker_train 

    def get_time_eval(self):
        return self.time_tracker_eval
    
    
class BaseProfiler(BaseTrainer):
    def train(self):
        # first wrap all relevant functions

        # wrap the estimate_loss function
        self.estimate_loss = TimeWrapper(self.estimate_loss)

        # wrap the _run_step function
        self._run_step = TimeWrapper(self._run_step)

        # wrap the save_model function
        self._save_model = TimeWrapper(self._save_model)

        # wrap the model forward pass
        self.model = TimeWrapperModel(self.model)

        # wrap the optimizer step
        self.optimizer.step = TimeWrapper(self.optimizer.step)

        # wrap the scheduler step
        self.scheduler.step = TimeWrapper(self.scheduler.step)

        # wrap the get_batch function
        self.dataloader.get_batch = TimeWrapper(self.dataloader.get_batch)


        # Start profiling
        # run training loop
        self.run_training_loop()

        # get the time for each function
        time_dict = {
            "estimate_loss": self.estimate_loss.get_time(),
            "_run_step": self._run_step.get_time(),
            "_save_model": self._save_model.get_time(),
            "model_train_pass": self.model.get_time_train(),
            "model_eval_pass": self.model.get_time_eval(),
            "optimizer.step": self.optimizer.step.get_time(),
            "scheduler.step": self.scheduler.step.get_time(),
            "dataloader.get_batch": self.dataloader.get_batch.get_time()
        }

        input(time_dict)
        
