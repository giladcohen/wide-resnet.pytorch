from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torch.autograd import Variable
from collections import OrderedDict

def criterion2(params):
    loss_conv   = 0.0
    loss_linear = 0.0
    l1_loss = torch.nn.L1Loss()
    for W in params:
        if W.requires_grad:
            if len(W.shape) == 4:  #conv weight
                sum_w = W.sum(-1).sum(-1)
                ideal_sum_w = torch.zeros_like(sum_w)
                loss_conv += l1_loss(sum_w, ideal_sum_w)
            if len(W.shape) == 2:  #linear weight
                sum_w = W.sum(-1)
                ideal_sum_w = torch.zeros_like(sum_w)
                loss_linear += l1_loss(sum_w, ideal_sum_w)
    return loss_conv + loss_linear

def criterion2_v1(params):
    weight_dict = OrderedDict()
    bias_dict   = OrderedDict()

    # collecting and arranging
    for W in params:
        # i+=1
        # print("{}) {}\t {}".format(i, W[0], W[1].shape))
        if W[1].requires_grad:
            if W[0].find("conv") != -1 or W[0].find("shortcut") != -1 or W[0].find("linear") != -1:
                split_str = W[0].split('.')
                param_str = split_str[0]
                for i in range(1, len(split_str)-1):
                    param_str += '.' + split_str[i]
                if "weight" in W[0]:
                    weight_dict[param_str] = W[1].data
                elif "bias" in W[0]:
                    bias_dict[param_str] = W[1].data
                else:
                    raise AssertionError("Expected weight or bias, but got W[0]={}".format(W[0]))
    # adding to loss
    loss = 0.0
    params_to_punish = weight_dict.keys()
    return loss
