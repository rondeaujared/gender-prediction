import abc
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src import LOG_DIR


class AbstractTrainer(metaclass=abc.ABCMeta):

    def __init__(self, model, tr_dl, val_dl, loss_fn,
                 optim, optim_kwargs,
                 scheduler=None, scheduler_kwargs=None,
                 dtype=torch.float32,
                 print_every_iters=50):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.savedir = f"{LOG_DIR}"
        self.e = 0
        self.train_step = 0
        self.eval_step = 0
        self.writer = SummaryWriter()

        self.model = model
        self.tr_dl = tr_dl
        self.val_dl = val_dl
        self.loss_fn = loss_fn

        self.dtype = dtype
        self.print_every_iters = print_every_iters

        self.model.to(device=self.device, dtype=self.dtype)
        self.optim = optim(self.model.parameters(), **optim_kwargs)
        if scheduler and scheduler_kwargs:
            self.scheduler = scheduler(self.optim, **scheduler_kwargs)

    def train(self, n_epochs: int):
        best_val = None
        best_fname = None
        for _ in range(n_epochs):
            self.model.train()
            tr_logs = self.epoch(dl=self.tr_dl, train=True)
            tr_elog = self.log_epoch(tr_logs)
            self.tensorboard_log(tr_elog, train=True)
            self.model.eval()
            with torch.no_grad():
                val_logs = self.epoch(dl=self.val_dl, train=False)
                val_elog = self.log_epoch(val_logs)
                self.tensorboard_log(val_elog, train=False)

                better, score = self.score_fn(val_elog, best_val)
                if better:
                    out = f"New best score: {score:.5f}"
                    if best_val: out += f" beats {best_val:.5f}"
                    print(out)
                    best_fname = self.save_weights(self.model, self.savedir)
                    best_val = score
            self.e += 1
        return best_val, best_fname

    def epoch(self, dl, train):
        logs = []
        for ix, (images, labels) in enumerate(dl):
            log = self.step(images, labels, train)
            logs.append(log)
        return logs

    def step(self, images, labels, train: bool):
        images = images.to(device=self.device, dtype=self.dtype)
        labels = labels.to(device=self.device)
        output = self.model(images)
        loss = self.loss_fn(output, labels)
        if train:
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            self.scheduler.step()
            self.train_step += 1

            if self.train_step % self.print_every_iters == 0:
                self.writer.add_scalar('loss/batch/train', loss.item(), self.train_step)
                if self.scheduler:
                    self.writer.add_scalar('scheduler lr', np.array(self.scheduler.get_last_lr()).mean(), self.train_step)
        else:
            self.eval_step += 1
            if self.eval_step % self.print_every_iters == 0:
                self.writer.add_scalar('loss/batch/valid', loss.item(), self.eval_step)
        with torch.no_grad():
            log = self.log_step(output, labels, loss.item())
        return log

    @abc.abstractmethod
    def log_step(self, output, labels, loss) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def log_epoch(self, logs) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def tensorboard_log(self, logs, train: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def score_fn(self, log, score) -> (bool, float):
        """
        :param log: Dict returned by log_epoch
        :param score: old best score
        :return: True if new best score is better than old, current score
        """
        raise NotImplementedError

    @staticmethod
    def save_weights(model, savedir, prefix='', suffix=''):
        time = datetime.datetime.now()
        s = f"{prefix}_{time.month}_{time.day}_{time.hour}_{time.minute}_{time.second}_{suffix}.pth"
        fname = f"{savedir}/{s}"
        print(f"Saving to: {fname}")
        torch.save(model.state_dict(), fname)
        return fname
