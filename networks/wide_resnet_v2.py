import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import sys
import numpy as np
from collections import OrderedDict

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, use_bn=True):
        super(wide_basic, self).__init__()
        self.use_bn = use_bn

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=not self.use_bn)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=not self.use_bn)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=not self.use_bn),
            )

    def forward(self, x):
        out = x
        if self.use_bn:
            out = self.bn1(out)
        out = self.dropout(self.conv1(F.relu(out)))
        if self.use_bn:
            out = self.bn2(out)
        out = self.conv2(F.relu(out))
        out += self.shortcut(x)

        return out

class Wide_ResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, use_bn):
        super(Wide_ResNet, self).__init__()
        self.e_net = OrderedDict()
        self.v_net = OrderedDict()
        self.in_planes = 16
        self.use_bn = use_bn

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        self.n = (depth-4)/6
        k = widen_factor
        self.dropout_rate = dropout_rate

        print('| Wide-Resnet %dx%d' %(depth, k))
        self.nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(3, self.nStages[0], bias=not self.use_bn)
        # self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        # self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        # self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)

        self.layer1_0 = wide_basic(self.nStages[0], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer1_1 = wide_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer1_2 = wide_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer1_3 = wide_basic(self.nStages[1], self.nStages[1], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.layer2_0 = wide_basic(self.nStages[1], self.nStages[2], self.dropout_rate, stride=2, use_bn=self.use_bn)
        self.layer2_1 = wide_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer2_2 = wide_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer2_3 = wide_basic(self.nStages[2], self.nStages[2], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.layer3_0 = wide_basic(self.nStages[2], self.nStages[3], self.dropout_rate, stride=2, use_bn=self.use_bn)
        self.layer3_1 = wide_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer3_2 = wide_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn)
        self.layer3_3 = wide_basic(self.nStages[3], self.nStages[3], self.dropout_rate, stride=1, use_bn=self.use_bn)

        self.bn1 = nn.BatchNorm2d(self.nStages[3], momentum=0.9)
        self.linear = nn.Linear(self.nStages[3], num_classes)

    # def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, dropout_rate, stride, self.use_bn))
    #         self.in_planes = planes
    #
    #     return nn.Sequential(*layers)

    def _record(self, name, x):
        """ Recording layer output's mean and standard deviation
        :param name: name to be recorded
        :param x: signal to record
        :return: None
        """
        x_1d = x.view(x.size(0), -1)
        self.e_net[name] = torch.mean(x_1d, dim=1)
        self.v_net[name] = torch.std(x_1d,  dim=1)

    def forward(self, x):
        self._record('image', x)
        out = self.conv1(x)
        self._record('conv1_out', out)

        out = self.layer1_0(out)
        self._record('layer1.0_out', out)
        out = self.layer1_1(out)
        self._record('layer1.1_out', out)
        out = self.layer1_2(out)
        self._record('layer1.2_out', out)
        out = self.layer1_3(out)
        self._record('layer1.3_out', out)

        out = self.layer2_0(out)
        self._record('layer2.0_out', out)
        out = self.layer2_1(out)
        self._record('layer2.1_out', out)
        out = self.layer2_2(out)
        self._record('layer2.2_out', out)
        out = self.layer2_3(out)
        self._record('layer2.3_out', out)

        out = self.layer3_0(out)
        self._record('layer3.0_out', out)
        out = self.layer3_1(out)
        self._record('layer3.1_out', out)
        out = self.layer3_2(out)
        self._record('layer3.2_out', out)
        out = self.layer3_3(out)
        self._record('layer3.3_out', out)

        if self.use_bn:
            out = self.bn1(out)
        out = F.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == '__main__':
    net=Wide_ResNet(28, 10, 0.3, 10)
    y = net(Variable(torch.randn(1,3,32,32)))

    print(y.size())
