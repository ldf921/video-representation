
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models

from .base import accuracy, time_distributed, time_distributed_apply, load_network
from .framework import Framework
from .lstm import SeqPredFramework


def stack_frames(frames, reverse=False):
    '''
    frames : [time_step, batch_size, channels, ...]
    output : [time_step - 1, batch_size, channels * 2, ...]
    '''
    T = frames.size(0)
    if reverse:
        return torch.cat([frames.narrow(0, 1, T - 1), frames.narrow(0, 0, T - 1)], dim=2)
    else:
        return torch.cat([frames.narrow(0, 0, T - 1), frames.narrow(0, 1, T - 1)], dim=2)


class StackCL(nn.Module):
    def __init__(self, classes, lstm_units, sync_bn=False, load_lstm=None, load_backbone=None):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(3 * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        if load_backbone is not None:
            load_network(resnet, load_backbone, 'module.backbone.')
        if sync_bn:
            print('Convert model using sync bn')
            resnet = convert_model(resnet)
        self.backbone = resnet

        self.lstm = nn.LSTM(in_planes, lstm_units)
        if load_lstm is not None:
            load_network(self.lstm, load_lstm, 'module.lstm.')
        self.fc = nn.Linear(lstm_units, classes)

    def forward(self, frames):
        self.lstm.flatten_parameters()
        frames = stack_frames(frames)
        features = time_distributed(self.backbone, frames)
        features, _ = self.lstm(features)
        return time_distributed(self.fc, features)


class StackCLOrder(nn.Module):
    def __init__(self, lstm_units, sync_bn=False, load_lstm=None, load_backbone=None):
        super().__init__()
        resnet = models.resnet50(pretrained=False)
        resnet.conv1 = nn.Conv2d(3 * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        if load_backbone is not None:
            load_network(resnet, load_backbone, 'module.backbone.')
        if sync_bn:
            print('Convert model using sync bn')
            resnet = convert_model(resnet)
        self.backbone = resnet

        self.lstm = nn.LSTM(in_planes, lstm_units)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, lstm_units))
        self.predict_mlp = nn.Sequential(
            nn.Linear(lstm_units * 2, lstm_units * 2),
            nn.ReLU(),
            nn.Linear(lstm_units * 2, 2)
            )

        self.register_buffer('label', torch.LongTensor([1, 0]))

    def forward(self, orig_frames):
        self.lstm.flatten_parameters()
        # stack sequence with length - 1
        frames = stack_frames(orig_frames)
        T, batch_size = frames.size()[:2]
        features = time_distributed(self.backbone, frames)
        lstm_features, _ = self.lstm(features.narrow(0, 0, T - 1))

        frames_r = stack_frames(orig_frames, reverse=True).narrow(0, 1, T - 1) # remove first pair
        features_r = time_distributed_apply(frames_r, self.backbone, self.fc)
        features = time_distributed_apply(features.narrow(0, 1, T - 1), self.fc) # remove first pair

        T = T - 1
        lstm_features = lstm_features.repeat(1, 1, 2).view(T * batch_size * 2, -1)
        diff_features = torch.cat([features, features_r], dim = -1).view(T * batch_size * 2, -1) # P, N, P, N, ...
        concat_features = torch.cat([lstm_features, diff_features], dim = -1)
        logits = self.predict_mlp(concat_features).view(T, batch_size * 2, 2)
        labels = self.label.repeat(T, batch_size, 1).view(T, batch_size * 2)

        return logits, labels


class StackCLPredFramework(SeqPredFramework):
    def build_network(self):
        self.model = StackCLOrder(**self.config['network'])

def stackcl_framework(config):
    if config['framework'] == 'stackcl-order':
        return StackCLPredFramework(config)




