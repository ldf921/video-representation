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
    """ Extract features from intermediate layers
        Return dim [F * N * input_dim]
        Last frame is in accurate due to looping, proceed with caution
    Args:
        model (nn.Module): The model in question, Resnet 50
    """
    # Transform input [F * N * C * H * W] -> [(F * N) * C * H * W]
    F, N, C, H, W = input.size()
    x = input.view(F * N, C, H, W)
    
    # In order to get intermidiate features, we cannot use forward hook since this gives parallel issues
    for i, part in enumerate(model.children()):
        x = part(x)
        # 3 features in total, each is [(F * N) * C_layer * H_layer * W_layer]
        # if i >= 5 and i <=7:
        if i == 6:
            # Shift output in F dimension, align each on with next frame
            _, C_layer, H_layer, W_layer = x.size()
            feature = x.view(F, N, C_layer, H_layer, W_layer)
            next_feature = feature.roll(-1, dims=0)
            # Sum over C dimension, view into [F * N * (H_layer * W_layer)]
            # correlations.append(torch.sum(feature * next_feature, dim=2).view(F, N, -1))
            x = feature * next_feature
            x = x.view(F*N, C_layer, H_layer, W_layer)
        elif i > 7:
            break


    # Cat and return
    return x.view(F, N, -1)


def one_kernel_feature_extraction(model: nn.Module, input: torch.tensor, D: int = 3, padding: int = 1, stride: int = 1) -> torch.tensor:
    """ Extract features from intermediate layers
        Return dim [F * N * input_dim]
        Last frame is in accurate due to looping, proceed with caution
    Args:
        model (nn.Module): The model in question, Resnet 50
    """
    # Transform input [F * N * C * H * W] -> [(F * N) * C * H * W]
    F, N, C, H, W = input.size()
    x = input.view(F * N, C, H, W)
    unfolder = nn.Unfold(kernel_size = D, padding=padding, stride=stride)
    # Compute correlation
    correlations = []
    # In order to get intermidiate features, we cannot use forward hook since this gives parallel issues
    for i, part in enumerate(model.children()):
        x = part(x)
        # 2 features in total, each is [(F * N) * C_layer * H_layer * W_layer]
        if i == 6:
            # Shift output in F dimension, align each on with next frame
            _, C_layer, H_layer, W_layer = x.size()
            feature = x.view(F, N, C_layer, H_layer, W_layer)
            next_feature = feature.roll(-1, dims=0)
            # Each feature perform inner product with a block of D^2
            feature = feature.unsqueeze(dim=3)
            feature = feature.repeat(1,1,1,D**2,1,1).view((F*N, -1, H_layer* W_layer))
            next_feature = unfolder(next_feature.view(F*N, C_layer, H_layer, W_layer))
            folder = nn.Fold(output_size = (H_layer, W_layer), kernel_size = D, padding = 1)
            # print(feature.size(), next_feature.size())
            x = folder(feature * next_feature)
            # Sum over C dimension, view into [F * N * (H_layer * W_layer)]
        elif i > 7:
            break

    # Cat and return
    return x.view(F, N, -1)

def next_two_feature_extraction(model: nn.Module, input: torch.tensor, D: int = 3, padding: int = 1, stride: int = 1) -> torch.tensor:
    """ Extract features from intermediate layers
        Return dim [F * N * input_dim]
        Last frame is in accurate due to looping, proceed with caution
    Args:
        model (nn.Module): The model in question, Resnet 50
    """
    # Transform input [F * N * C * H * W] -> [(F * N) * C * H * W]
    F, N, C, H, W = input.size()
    x = input.view(F * N, C, H, W)
    unfolder = nn.Unfold(kernel_size = D, padding=padding, stride=stride)

    # In order to get intermidiate features, we cannot use forward hook since this gives parallel issues
    for i, part in enumerate(model.children()):
        x = part(x)
        # 2 features in total, each is [(F * N) * C_layer * H_layer * W_layer]
        if i == 6:
            # Shift output in F dimension, align each on with next frame
            _, C_layer, H_layer, W_layer = x.size()
            feature = x.view(F, N, C_layer, H_layer, W_layer)
            next_feature = feature.roll(-1, dims=0).view(F*N, C_layer, H_layer, W_layer)
            skipped_feature = feature.roll(-2, dims=0).view(F*N, C_layer, H_layer, W_layer)
            next_two_features = torch.cat([next_feature, skipped_feature])
            # Each feature perform inner product with a block of D^2
            feature = feature.unsqueeze(dim=3)
            feature = feature.repeat(1,1,1,D**2,1,1).view(F*N, -1, H_layer* W_layer)
            feature = feature.repeat(2,1,1)
            next_two_features = unfolder(next_two_features)
            next_two_features = feature * next_two_features
        elif i  == 8:
            # Return tensor [2FN * D^2 * H * W], first FN is next_feature, second FN is skipped_feature
            return torch.sum(next_two_features.view(2*F*N, C_layer, D**2, H_layer, W_layer), dim=1), x, F, N

def next_feature_extraction(model: nn.Module, input: torch.tensor, D: int = 3, padding: int = 1, stride: int = 1) -> torch.tensor:
    """ Extract features from intermediate layers
        Return dim [F * N * input_dim]
        Last frame is in accurate due to looping, proceed with caution
    Args:
        model (nn.Module): The model in question, Resnet 50
    """
    # Transform input [F * N * C * H * W] -> [(F * N) * C * H * W]
    F, N, C, H, W = input.size()
    x = input.view(F * N, C, H, W)
    unfolder = nn.Unfold(kernel_size = D, padding=padding, stride=stride)

    # In order to get intermidiate features, we cannot use forward hook since this gives parallel issues
    next_feature = None
    for i, part in enumerate(model.children()):
        x = part(x)
        if i == 6:
            # Shift output in F dimension, align each on with next frame
            _, C_layer, H_layer, W_layer = x.size()
            feature = x.view(F, N, C_layer, H_layer, W_layer)
            next_feature = feature.roll(-1, dims=0).view(F*N, C_layer, H_layer, W_layer)
            # Each feature perform inner product with a block of D^2
            unfold_feature = feature.unsqueeze(dim=3)
            unfold_feature = unfold_feature.repeat(1,1,1,D**2,1,1).view(F*N, -1, H_layer* W_layer)
            next_feature = unfolder(next_feature)
            next_feature = unfold_feature * next_feature
            # Return tensor [FN * D^2 * H * W]
            next_feature = torch.sum(next_feature.view(F*N, C_layer, D**2, H_layer, W_layer), dim=1)
        elif i == 8:
            return next_feature, x,  F, N
