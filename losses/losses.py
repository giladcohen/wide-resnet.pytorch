from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import torch
from torch.autograd import Variable
from collections import OrderedDict
import numpy as np

def criterion1(net):
    e_net = net.e_net
    v_net = net.v_net
    l1_loss = torch.nn.L1Loss(size_average=True)  #TODO(gilad): try also size_average=False

    E_loss = OrderedDict()
    V_loss = OrderedDict()

    all_keys = e_net.keys()
    for i, key in enumerate(all_keys):
        if key is 'image':
            # don't calculate loss for the input
            assert i == 0
            continue

        # E loss
        e1 = e_net[all_keys[i-1]]
        e2 = e_net[all_keys[i]]
        e_diff = e2-e1
        ideal_e = torch.zeros_like(e_diff)
        E_loss[key] = l1_loss(e_diff, ideal_e)

        # V loss
        v1 = v_net[all_keys[i-1]]
        v2 = v_net[all_keys[i]]
        v_diff = v2-v1
        ideal_v = torch.zeros_like(v_diff)
        V_loss[key] = l1_loss(v_diff, ideal_v)

    return E_loss, V_loss

def criterion2_v0(params):
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

def criterion2(params):
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
                    weight_dict[param_str] = W[1].clone()
                elif "bias" in W[0]:
                    bias_dict[param_str] = W[1].clone()
                else:
                    raise AssertionError("Expected weight or bias, but got W[0]={}".format(W[0]))

    l1_loss = torch.nn.L1Loss(size_average=False)
    # adding to E loss
    E_loss = 0.0
    for name in weight_dict.keys():
        if name.find("linear") != -1:
            sum_w = weight_dict[name].sum(-1)
        else:
            sum_w = weight_dict[name].sum(-1).sum(-1).sum(-1)
        sum_w_b = sum_w + bias_dict[name]
        ideal_sum_w_b = torch.zeros_like(sum_w_b)
        E_loss += l1_loss(sum_w_b, ideal_sum_w_b)

    # adding to V loss
    V_loss = 0.0
    ideal_var = np.sqrt(2*np.pi/(np.pi-1))
    for name in weight_dict.keys():
        w = weight_dict[name].view(weight_dict[name].size(0), -1)
        if name.find("linear") != -1:
            l2norm_w = w.norm(2, dim=-1)
        else:
            l2norm_w = w.norm(2, dim=-1)
        ideal_var_w = ideal_var * torch.ones_like(l2norm_w)
        V_loss += l1_loss(l2norm_w, ideal_var_w)
    return E_loss, V_loss

def rescale_loss_dict(loss_dict, rescale_dict):
    """
    :param loss_dict: dictionary of losses
    :param rescale_dict: dictionaries of weight scaling with the same keys
    :return: None. Updates the loss_dict
    """
    all_keys = loss_dict.keys()
    for key in all_keys:
        loss_dict[key] *= rescale_dict[key]
