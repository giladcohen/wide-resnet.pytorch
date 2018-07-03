from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

def print_model_parameters(model):
    """Print all model parameters which require gradient"""
    i = 0
    for name, param in model.named_parameters():
        i += 1
        if param.requires_grad:
            print("{}) {}\t {}".format(i, name, param.shape))

