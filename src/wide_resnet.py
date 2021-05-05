# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    'wide_resnet28w2'
]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride=1
    ):
        super(WideBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):

    def __init__(
        self,
        depth,
        widen_factor,
        dropout_rate=0.3
    ):
        super(WideResNet, self).__init__()

        self.in_planes = 16
        assert ((depth-4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)/6
        k = widen_factor

        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, nStages[0])
        self.layer1 = self._wide_layer(WideBasicBlock, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(WideBasicBlock, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(WideBasicBlock, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.99)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = None
        self.pred = None

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def _forward_backbone(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.relu(self.bn1(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)
        return x

    def _forward_head(self, x):
        if self.pred is not None:
            x = self.pred(x)
        return x

    def forward(self, inputs, return_before_head=False):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _h = self._forward_backbone(torch.cat(inputs[start_idx:end_idx]))
            _z = self._forward_head(_h)
            if start_idx == 0:
                h, z = _h, _z
            else:
                h, z = torch.cat((h, _h)), torch.cat((z, _z))
            start_idx = end_idx

        if return_before_head:
            return h, z

        return z


def wide_resnet28w2(**kwargs):
    encoder = WideResNet(28, 2, **kwargs)
    return encoder
