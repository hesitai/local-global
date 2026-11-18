"""Delayed model EMA and portable training state helpers."""
from copy import deepcopy
import random
import numpy as np
import torch


class ModelEMA:
    def __init__(self, decay=0.999, start_epoch=4):
        if not 0 <= decay < 1 or start_epoch < 1:
            raise ValueError('Invalid EMA configuration')
        self.decay, self.start_epoch = decay, start_epoch
        self.model = None
        self.num_updates = 0

    def begin_epoch(self, model, epoch):
        if epoch >= self.start_epoch and self.model is None:
            self.model = deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step_succeeded=True):
        if self.model is None or not step_succeeded:
            return
        source_state = model.state_dict()
        for key, target in self.model.state_dict().items():
            source = source_state[key].detach()
            if target.is_floating_point() or target.is_complex():
                target.mul_(self.decay).add_(source, alpha=1 - self.decay)
            else:
                target.copy_(source)
        self.num_updates += 1

    def state_dict(self):
        return dict(decay=self.decay, start_epoch=self.start_epoch,
                    num_updates=self.num_updates,
                    state_dict=None if self.model is None else self.model.state_dict())

    def load_state_dict(self, state, model):
        if (state['decay'], state['start_epoch']) != (self.decay, self.start_epoch):
            raise ValueError('Resume EMA configuration mismatch')
        self.num_updates = state['num_updates']
        if state['state_dict'] is not None:
            self.begin_epoch(model, self.start_epoch)
            self.model.load_state_dict(state['state_dict'], strict=True)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def scheduler_state(scheduler):
    state = scheduler.state_dict()
    after = state.pop('after_scheduler', None)
    return dict(main=state, after=None if after is None else after.state_dict())


def restore_scheduler(scheduler, state):
    scheduler.load_state_dict(state['main'])
    if state['after'] is not None:
        scheduler.after_scheduler.load_state_dict(state['after'])
