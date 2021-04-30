# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch

from src.utils import AverageMeter


class LARS(torch.optim.Optimizer):

    def __init__(self, optimizer, trust_coefficient=0.02, eps=1e-8):
        self.optim = optimizer
        self.trust_coefficient = trust_coefficient
        self.eps = eps

    def __getstate__(self):
        return self.optim.__getstate__()

    def __setstate__(self, state):
        self.optim.__setstate__(state)

    @property
    def state(self):
        return self.optim.state

    def __repr__(self):
        return self.optim.__repr__()

    @property
    def param_groups(self):
        return self.optim.param_groups

    @param_groups.setter
    def param_groups(self, value):
        self.optim.param_groups = value

    def state_dict(self):
        return self.optim.state_dict()

    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)

    def zero_grad(self):
        self.optim.zero_grad()

    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)

    def step(self):
        with torch.no_grad():
            stats = AverageMeter()
            weight_decays = []
            for group in self.optim.param_groups:

                # -- takes weight decay control from wrapped optimizer
                weight_decay = group['weight_decay'] if 'weight_decay' in group else 0
                weight_decays.append(weight_decay)

                # -- user wants to exclude this parameter group from LARS
                #    adaptation
                if ('LARS_exclude' in group) and group['LARS_exclude']:
                    continue
                group['weight_decay'] = 0

                for p in group['params']:
                    if p.grad is None:
                        continue
                    param_norm = torch.norm(p.data)
                    grad_norm = torch.norm(p.grad.data)

                    if param_norm != 0 and grad_norm != 0:
                        adaptive_lr = self.trust_coefficient * (param_norm) / (grad_norm + param_norm * weight_decay + self.eps)

                        stats.update(adaptive_lr)
                        p.grad.data += weight_decay * p.data
                        p.grad.data *= adaptive_lr

        self.optim.step()
        # -- return weight decay control to wrapped optimizer
        for i, group in enumerate(self.optim.param_groups):
            group['weight_decay'] = weight_decays[i]

        return stats
