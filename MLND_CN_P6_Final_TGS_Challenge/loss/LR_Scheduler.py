import datetime
import math
import os
import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import types
from torch._six import inf
from functools import partial, wraps
import warnings


class CosineAnnealingLR(_LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing
    schedule, where :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:

    .. math::

        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 +
        \cos(\frac{T_{cur}}{T_{max}}\pi))

    When last_epoch=-1, sets initial lr as lr.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. This implementation
    contains restarts and T_mul.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(self, optimizer, T_max, T_mul, lr_min=0, last_epoch=-1, val_mode='max', save_snapshots=False):
        self.T_max = T_max
        self.T_mul = T_mul
        self.T_curr = 0
        self.lr_min = lr_min
        self.save_snapshots = save_snapshots
        self.val_mode = val_mode
        self.best_model_path = None
        self.reset = 0
        self.lr_history = []

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        r = self.T_curr % self.T_max

        if not r and self.last_epoch > 0:
            self.T_max *= self.T_mul
            self.T_curr = 1
            self.update_saving_vars()
        else:
            self.T_curr += 1
            

        new_lrs = [self.lr_min + (base_lr - self.lr_min) * 
                    (1 + math.cos(math.pi * r / self.T_max)) / 2 
                    for base_lr in self.base_lrs]
        self.lr_history.append(new_lrs)
        return new_lrs

    def step(self, epoch=None, save_dict=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.save_snapshots and save_dict is not None:
            self.save_best_model(save_dict)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def update_saving_vars(self):
        self.reset += 1
        self.best_model_path = None

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf


    def save_best_model(self, save_dict):
        metric = save_dict['metric']
        fold = save_dict['fold']
        save_dir = save_dict['save_dir']
        state_dict = save_dict['state_dict']

        if (self.val_mode == 'max' and metric > self.best_metric) or (
                self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if fold is not None:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Fold{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, fold, self.last_epoch, self.reset, metric))
            else:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Epoach{}_reset{:}_val{:.3f}'.format(date, self.last_epoch, self.reset, metric))

            torch.save(state_dict, self.best_model_path)

class FindLR(_LRScheduler):
    """
    inspired by fast.ai @https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
    """
    def __init__(self, optimizer, max_steps, max_lr=10):
        self.max_steps = max_steps
        self.max_lr = max_lr
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * ((self.max_lr / base_lr) ** (self.last_epoch / (self.max_steps - 1)))
                for base_lr in self.base_lrs]


class NoamLR(_LRScheduler):
    """
    Implements the Noam Learning rate schedule. This corresponds to increasing the learning rate
    linearly for the first ``warmup_steps`` training steps, and decreasing it thereafter proportionally
    to the inverse square root of the step number, scaled by the inverse square root of the
    dimensionality of the model. Time will tell if this is just madness or it's actually important.
    Parameters
    ----------
    warmup_steps: ``int``, required.
        The number of steps to linearly increase the learning rate.
    """
    def __init__(self, optimizer, warmup_steps):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        last_epoch = max(1, self.last_epoch)
        scale = self.warmup_steps ** 0.5 * min(last_epoch ** (-0.5), last_epoch * self.warmup_steps ** (-1.5))
        return [base_lr * scale for base_lr in self.base_lrs]


class Poly(_LRScheduler):
    def __init__(self, optimizer, num_epochs, iters_per_epoch=0, warmup_epochs=0, last_epoch=-1, val_mode='max', save_snapshots=False):
        self.iters_per_epoch = iters_per_epoch
        self.cur_iter = 0
        self.N = num_epochs * iters_per_epoch
        self.warmup_iters = warmup_epochs * iters_per_epoch
        super(Poly, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        T = self.last_epoch * self.iters_per_epoch + self.cur_iter
        factor =  pow((1 - 1.0 * T / self.N), 0.9)
        if self.warmup_iters > 0 and T < self.warmup_iters:
            factor = 1.0 * T / self.warmup_iters

        self.cur_iter %= self.iters_per_epoch
        self.cur_iter += 1
        return [base_lr * factor for base_lr in self.base_lrs]



class CyclicLR(_LRScheduler):
    """Sets the learning rate of each parameter group according to
    cyclical learning rate policy (CLR). The policy cycles the learning
    rate between two boundaries with a constant frequency, as detailed in
    the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.
    This class has three built-in policies, as put forth in the paper:
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    This implementation was adapted from the github repo: `bckenstler/CLR`_
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        base_lr (float or list): Initial learning rate which is the
            lower boundary in the cycle for each parameter group.
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size_up (int): Number of training iterations in the
            increasing half of a cycle. Default: 2000
        step_size_down (int): Number of training iterations in the
            decreasing half of a cycle. If step_size_down is None,
            it is set to step_size_up. Default: None
        mode (str): One of {triangular, triangular2, exp_range}.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
            Default: 'triangular'
        gamma (float): Constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
            Default: 1.0
        scale_fn (function): Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            If specified, then 'mode' is ignored.
            Default: None
        scale_mode (str): {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle).
            Default: 'cycle'
        cycle_momentum (bool): If ``True``, momentum is cycled inversely
            to learning rate between 'base_momentum' and 'max_momentum'.
            Default: True
        base_momentum (float or list): Lower momentum boundaries in the cycle
            for each parameter group. Note that momentum is cycled inversely
            to learning rate; at the peak of a cycle, momentum is
            'base_momentum' and learning rate is 'max_lr'.
            Default: 0.8
        max_momentum (float or list): Upper momentum boundaries in the cycle
            for each parameter group. Functionally,
            it defines the cycle amplitude (max_momentum - base_momentum).
            The momentum at any cycle is the difference of max_momentum
            and some scaling of the amplitude; therefore
            base_momentum may not actually be reached depending on
            scaling function. Note that momentum is cycled inversely
            to learning rate; at the start of a cycle, momentum is 'max_momentum'
            and learning rate is 'base_lr'
            Default: 0.9
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
    Example:
         optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
         scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
         data_loader = torch.utils.data.DataLoader(...)
         for epoch in range(10):
             for batch in data_loader:
                 train_batch(...)
                 scheduler.step()
    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(self,
                    optimizer,
                    base_lr,
                    max_lr,
                    step_size_up=100,
                    step_size_down=None,
                    mode='triangular',
                    gamma=1.,
                    scale_fn=None,
                    scale_mode='cycle',
                    cycle_momentum=True,
                    base_momentum=0.85,
                    max_momentum=0.95,
                    last_epoch=-1,
                    val_mode='max', 
                    save_snapshots=False):

        self.optimizer = optimizer

        base_lrs = self._format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = self._format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        if mode not in ['triangular', 'triangular2', 'exp_range'] \
                and scale_fn is None:
            raise ValueError('mode is invalid and scale_fn is None')

        self.mode = mode
        self.gamma = gamma
        self.save_snapshots = save_snapshots
        self.val_mode = val_mode
        self.best_model_path = None

        if self.val_mode == 'max':
            self.best_metric = -np.inf
        elif self.val_mode == 'min':
            self.best_metric = np.inf

        if scale_fn is None:
            if self.mode == 'triangular':
                self.scale_fn = self._triangular_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = self._triangular2_scale_fn
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = self._exp_range_scale_fn
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = self._format_param('base_momentum', optimizer, base_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))
            self.max_momentums = self._format_param('max_momentum', optimizer, max_momentum)

        super(CyclicLR, self).__init__(optimizer, last_epoch)

    def _format_param(self, name, optimizer, param):
        """Return correctly formatted lr/momentum for each param group."""
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError("expected {} values for {}, got {}".format(
                    len(optimizer.param_groups), name, len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def _triangular_scale_fn(self, x):
        return 1.

    def _triangular2_scale_fn(self, x):
        return 1 / (2. ** (x - 1))

    def _exp_range_scale_fn(self, x):
        return self.gamma**(x)

    def get_lr(self):
        """Calculates the learning rate at batch index. This function treats
        `self.last_epoch` as the last batch index.
        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            if self.scale_mode == 'cycle':
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        if self.cycle_momentum:
            momentums = []
            for base_momentum, max_momentum in zip(self.base_momentums, self.max_momentums):
                base_height = (max_momentum - base_momentum) * scale_factor
                if self.scale_mode == 'cycle':
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(self.last_epoch)
                momentums.append(momentum)
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

        return lrs

    def step(self, epoch=None, save_dict=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.save_snapshots and save_dict is not None:
            self.save_best_model(save_dict)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
            
    def save_best_model(self, save_dict):
        metric = save_dict['metric']
        fold = save_dict['fold']
        save_dir = save_dict['save_dir']
        state_dict = save_dict['state_dict']

        if (self.val_mode == 'max' and metric > self.best_metric) or (
                self.val_mode == 'min' and metric < self.best_metric):
            # Update best metric
            self.best_metric = metric
            # Remove old file
            if self.best_model_path is not None:
                os.remove(self.best_model_path)
            # Save new best model weights
            date = ':'.join(str(datetime.datetime.now()).split(':')[:2])
            if fold is not None:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Fold{:}_Epoach{}_val{:.3f}'.format(date, fold, self.last_epoch, metric))
            else:
                self.best_model_path = os.path.join(
                    save_dir,
                    '{:}_Epoach{}_val{:.3f}'.format(date, self.last_epoch, metric))

            torch.save(state_dict, self.best_model_path)

            


