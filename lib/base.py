import os
import torch
from torch import nn

def time_distributed(module, tensor):
    size = tensor.size()
    output = tensor.view((size[0] * size[1],) + size[2:])
    output = module(output)
    return output.view((size[0], size[1]) + output.size()[1:])

def time_distributed_apply(tensor, *modules):
    size = tensor.size()
    output = tensor.view((size[0] * size[1],) + size[2:])
    for layer in modules:
        output = layer(output)
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
    print('Loading checkpoint ', path)
    state_dict = torch.load(path)
    from collections import OrderedDict
    args_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith(name):
            args_dict[k.replace(name, '')] = v
    net.load_state_dict(args_dict)

def feature_extraction(model: nn.Module, input: torch.tensor) -> torch.tensor:
    """ Extract features from intermediate layers using register_forward_hook
        Return dim [F * N * input_dim]
        Last frame is in accurate due to looping, proceed with caution
    Args:
        model (nn.Module): The model in question, Resnet 50
    """
    # Transform input [F * N * C * H * W] -> [(F * N) * C * H * W]
    F, N, C, H, W = input.size()
    input = input.view(F * N, C, H, W)
    # 3 features in total, each is [(F * N) * C_layer * H_layer * W_layer]
    features = []
    # Hook function for register_forward_hook
    def hook(module, input, output):
        _, C_layer, H_layer, W_layer = output.size()
        features.append(output.view(F, N, C_layer, H_layer, W_layer))
    # Hook it up
    model.layer2[-1].register_forward_hook(hook)
    model.layer3[-1].register_forward_hook(hook)
    model.layer4[-1].register_forward_hook(hook)
    model(input)
    # Compute correlation
    correlations = []
    for feature in features:
        # Shift output in F dimension, align each on with next frame
        next_feature = feature.roll(-1, dims=0)
        # Sum over C dimension, view into [F * N * (H_layer * W_layer)]
        correlations.append(torch.tensordot(feature, next_feature, dims=2).view(F, N, -1))
    # Stack and return
    return torch.stack(correlations, dim=2)
