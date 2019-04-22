
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models

from .base import accuracy, time_distributed
from .framework import Framework
from .video_data import dataset, sampler, transforms


class LSTM(nn.Module):
    def __init__(self, features, lstm_units, diff=False):
        super().__init__()
        self.lstm = nn.LSTM(features, lstm_units)
        self.fc = nn.Linear(lstm_units, features)
        self.diff = diff

    def forward(self, features):
        self.lstm.flatten_parameters()
        out_features, _ = self.lstm(features)
        out_features = time_distributed(self.fc, out_features)
        if self.diff:
            return out_features + features
        else:
            return out_features


class ConvLSTM(nn.Module):
    def __init__(self, lstm_units, train_backbone = False):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.train_backbone = train_backbone
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.backbone = resnet

        self.lstm = nn.LSTM(in_planes, lstm_units)
        self.fc = nn.Linear(lstm_units, in_planes)

    def train(self, mode=True):
        self.lstm.train(mode)
        self.fc.train(mode)
        self.backbone.train(self.train_backbone and mode)
        if not self.train_backbone and mode:
            print('Fixing resnet in evaluation mode')

    def forward(self, frames):
        steps = features.size(0)
        batch_size = features.size(1)
        self.lstm.flatten_parameters()

        features = time_distributed(self.backbone(frames))
        out_features, _ = self.lstm(features.narrow(0, 0, steps - 1))
        out_features = self.fc(out_features.view((steps - 1) * batch_size, -1)).view(steps - 1, batch_size, -1)
        return out_features, features


class FramePredict(nn.Module):
    def __init__(self, lstm_units, task, train_backbone = False):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.train_backbone = train_backbone
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.backbone = resnet

        self.task = task
        if self.task == 'order':
            self.lstm = nn.LSTM(in_planes, lstm_units)
            self.compare_fc = nn.Sequential(
                nn.Linear(in_planes * 2, lstm_units),
                nn.ReLU())
            self.predict_mlp = nn.Sequential(
                nn.Linear(lstm_units * 2, 100),
                nn.ReLU(),
                nn.Linear(100, 2)
                )
        elif self.task == 'order-frame':
            self.compare_fc = nn.Sequential(
                nn.Linear(in_planes * 2, lstm_units),
                nn.ReLU(),
                nn.Linear(lstm_units, lstm_units),
                nn.ReLU(),
                nn.Linear(lstm_units, 2)
                )

        self.register_buffer('label', torch.LongTensor([1, 0]))

    def train(self, mode=True):
        for layer in self.children():
            if layer == self.backbone:
                layer.train(self.train_backbone and mode)
                if not self.train_backbone and mode:
                    print('Fixing resnet in evaluation mode')
            else:
                layer.train(mode)

    def forward(self, frames):
        steps = frames.size(0)
        batch_size = frames.size(1)

        features = time_distributed(self.backbone, frames)

        if self.task == 'order':
            T = steps - 2
            self.lstm.flatten_parameters()
            lstm_features, _ = self.lstm(features.narrow(0, 0, T))

            features_p = torch.cat([features.narrow(0, 1, T), features.narrow(0, 2, T)], dim = -1)
            features_n = torch.cat([features.narrow(0, 2, T), features.narrow(0, 1, T)], dim = -1)

            concat_features = torch.cat([features_p, features_n], dim = -1).view(T * batch_size * 2, -1)

            lstm_features = lstm_features.repeat(1, 1, 2).view(T * batch_size * 2, -1)
            fuse = torch.cat([lstm_features, self.compare_fc(concat_features)], dim = -1)
            predictions = self.predict_mlp(fuse).view(T, batch_size * 2, 2)

            d = frames.device
            labels = self.label.repeat(T, batch_size, 1).view(T, batch_size * 2)
            return predictions, labels
        elif self.task == 'order-frame':
            T = 1
            features_p = torch.cat([features.narrow(0, 0, T), features.narrow(0, 1, T)], dim = -1)
            features_n = torch.cat([features.narrow(0, 1, T), features.narrow(0, 0, T)], dim = -1)
            concat_features = torch.cat([features_p, features_n], dim = -1).view(T * batch_size * 2, -1)
            predictions = self.compare_fc(concat_features).view(T, batch_size * 2, 2)
            labels = self.label.repeat(T, batch_size, 1).view(T, batch_size * 2)
            return predictions, labels



def l2loss(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def build_dataset(split, shuffle, config):
    length = config['data']['length']
    t=transforms.Compose([
       transforms.CenterCrop((224, 224)),
       transforms.ToTensor()
       ])
    feature = config['data'].get('feature', True)
    data = dataset.build_kinetics_dataset(
            split,
            transforms = t,
            length = length,
            stride = 5,
            feature = feature
            )
    if shuffle:
        s = sampler.RandomSampler(data, loop=config['data']['loop'])
    else:
        s = sampler.ValSampler(data, stride=1)

    loader = torch.utils.data.DataLoader(
            data,
            sampler=s,
            batch_size=config['batch_size'],
            num_workers=config['num_worker']
            )
    return data, loader


class SeqFramework(Framework):
    def build_network(self):
        if self.config['data'].get('feature', True):
            self.feature = True
            self.model = LSTM(2048, **self.config['network'])
        else:
            self.feature = False
            self.model = ConvLSTM(**self.config['network'])

    def build_dataset(self):
        v = self.config['data']['version']
        self.train_data, self.train_loader = build_dataset('train_' + v, True, self.config)
        self.val_data, self.val_loader = build_dataset('valid_' + v, True, self.config)

    def train_batch(self, features, labels):
        steps = features.size(0)
        if self.feature:
            reconstr_features = self.model(features.narrow(0, 0, steps - 1))
        else:
            reconstr_features, features = self.model(features)
        losses = l2loss(reconstr_features, features.narrow(0, 1, steps - 1))
        naive_loss = torch.mean(l2loss(features.narrow(0, 0, steps - 1),
            features.narrow(0, 1, steps - 1)))
        loss = torch.mean(losses)
        return dict(loss=loss, naive=naive_loss)

    def eval_batch(self, features, labels):
        return self.train_batch(features, labels)

    def valid(self):
        return self.evaluate(self.val_loader)


class SeqPredFramework(Framework):
    def build_network(self):
        self.model = FramePredict(**self.config['network'])

    def build_dataset(self):
        v = self.config['data']['version']
        self.train_data, self.train_loader = build_dataset('train_' + v, True, self.config)
        self.val_data, self.val_loader = build_dataset('valid_' + v, True, self.config)

    def train_batch(self, features, labels):
        steps = features.size(0)
        logits, labels = self.model(features)
        logits = logits.view(-1, logits.size(2))
        labels = labels.view(-1)
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels)
        return dict(loss=loss, acc=acc)

    def eval_batch(self, features, labels):
        return self.train_batch(features, labels)

    def valid(self):
        return self.evaluate(self.val_loader)
