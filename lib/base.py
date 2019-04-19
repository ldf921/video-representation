import os
import torch
from torch import nn

def time_distributed(module, tensor):
    size = tensor.size()
    output = module(tensor.view((size[0] * size[1],) + size[2:]))
    return output.view((size[0], size[1]) + output.size()[1:])

def accuracy(scores, labels, weight=None):
    with torch.no_grad():
        _, pred = torch.max(scores, dim=-1)
        if weight is None:
            return torch.mean((labels == pred).float())
        else:
            return torch.sum((labels == pred).float() * weight)


# CHECKPOINTS_PREFIX = '/checkpoints/'
def load_network(net, path, name='module.'):
    # if not os.path.exists(path):
    #     path = os.path.join(CHECKPOINTS_PREFIX, path)
    state_dict = torch.load(path)
    from collections import OrderedDict
    args_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(name):
            args_dict[k.replace(name, '')] = v
    net.load_state_dict(args_dict)
