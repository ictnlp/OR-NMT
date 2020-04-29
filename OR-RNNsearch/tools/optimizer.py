from __future__ import division

import math
import torch.nn as nn
import torch.optim as opt
from torch.nn.utils import clip_grad_norm_

import wargs
from .utils import wlog

class Optim(object):

    def __init__(self, opt_mode, learning_rate, max_grad_norm):

        self.opt_mode = opt_mode
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.n_current_steps = 0
        self.warmup_steps = wargs.warmup_steps

        if wargs.lr_update_way == 'invsqrt':
            self.warmup_end_lr = learning_rate
            self.warmup_init_lr = wargs.warmup_init_lr if wargs.warmup_init_lr >= 0 else self.warmup_end_lr
            # linearly warmup for the first args.warmup_updates
            self.lr_step = (self.warmup_end_lr - self.warmup_init_lr) / wargs.warmup_steps
            # then, decay prop. to the inverse square root of the update number
            self.decay_factor = self.warmup_end_lr * ( wargs.warmup_steps**0.5 )
            # initial learning rate
            self.learning_rate = self.warmup_init_lr

    def __repr__(self):

        rst =  '\nOptimizer: {}\nLearning rate: {}\nGrad norm: {}\nwarmup steps: {} ' \
                '\ncurrent steps: {}\n'.format(self.opt_mode, self.learning_rate, self.max_grad_norm,
                    self.warmup_steps, self.n_current_steps)
        if wargs.lr_update_way == 'invsqrt':
            rst += 'warmup init lr: {}\nlr step: {}\ndecay factor: {}\nend lr: {}'.format(
                self.warmup_init_lr, self.lr_step, self.decay_factor, self.warmup_end_lr)

        return rst

    def init_optimizer(self, params):

        # careful: params may be a generator
        self.params = list(params)
        self.params = filter(lambda p: p.requires_grad, self.params)
        self.params = list(self.params)

        wlog('Init Optimizer ... ', 0)
        if self.opt_mode == 'sgd':
            self.optimizer = opt.SGD(self.params, lr=self.learning_rate)
            wlog('SGD ... lr: {}'.format(self.learning_rate))
        elif self.opt_mode == 'adagrad':
            self.optimizer = opt.Adagrad(self.params, lr=self.learning_rate)
            wlog('Adagrad ... lr: {}'.format(self.learning_rate))
        elif self.opt_mode == 'adadelta':
            self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=wargs.rho)
            #self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=0.95, eps=10e-06)
            #self.optimizer = opt.Adadelta(self.params, lr=self.learning_rate, rho=0.95, weight_decay=10e-5)
            wlog('Adadelta ... lr: {}, rho: {}'.format(self.learning_rate, wargs.rho))
        elif self.opt_mode == 'adam':
            self.optimizer = opt.Adam(self.params, lr=self.learning_rate,
                                      betas=[wargs.beta_1, wargs.beta_2], eps=wargs.adam_epsilon)
            wlog('Adam ... lr: {}, [ beta_1: {}, beta_2: {} ], adam_epsilon: {}'.format(
                self.learning_rate, wargs.beta_1, wargs.beta_2, wargs.adam_epsilon))
        else:
            wlog('Do not support this opt_mode {}'.format(self.opt_mode))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def step(self):

        # update the learning rate
        self.n_current_steps += 1
        if wargs.lr_update_way == 'noam':
            factor = ( wargs.d_model ** (-0.5) ) * min(
                (self.n_current_steps + 1) ** (-0.5),
                (self.n_current_steps + 1) * ( self.warmup_steps ** (-1.5) )
            )
            #self.learning_rate = factor
            #wlog('lrate = {}'.format(factor))
            self.learning_rate = wargs.learning_rate * factor
        elif wargs.lr_update_way == 'chen':
            n, s, e = wargs.n_co_models, wargs.s_step_decay, wargs.e_step_decay
            factor = min( 1 + ( self.n_current_steps * (n - 1) ) / ( n * self.warmup_steps ),
                         n,
                         n * ( (2 * n) ** ( ( s - n * self.n_current_steps ) / ( e - s ) ) ) )
            self.learning_rate = wargs.learning_rate * factor
        elif wargs.lr_update_way == 'invsqrt':
            if self.n_current_steps < wargs.warmup_steps:
                self.learning_rate = self.warmup_init_lr + self.n_current_steps*self.lr_step
            else:
                self.learning_rate = self.decay_factor * ( self.n_current_steps ** (-0.5) )

        #wlog('lr0 * factor = {} * {} = {}'.format(wargs.learning_rate, factor, self.learning_rate))
        #wlog('lr = {}'.format(self.learning_rate))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        '''
        param_groups:
        [{'betas': (0.9, 0.999),
          'eps': 1e-08,
          'lr': 0.0001,
          'params': [Variable containing:
           -0.7941 -0.9056 -0.1569
           -0.7084  1.7447 -0.6319
           [torch.FloatTensor of size 2x3], Variable containing:
           -1.0234 -0.2506 -0.3016  0.7835
            0.1354 -1.1608 -0.7858  0.2127
           -0.6725 -0.8482 -0.6999  1.5561
           [torch.FloatTensor of size 3x4]],
          'weight_decay': 0}]
        '''

        # clip by the gradients norm
        if self.max_grad_norm > 0.:
            #wlog('L2 norm Grad clip ... {}'.format(self.max_grad_norm))
            clip_grad_norm_(self.params, max_norm=self.max_grad_norm)

        self.optimizer.step()


