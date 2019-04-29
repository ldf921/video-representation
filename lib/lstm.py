import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models

from .base import accuracy, time_distributed, feature_extraction, one_kernel_feature_extraction
from .framework import Framework
from .video_data import dataset, sampler, transforms
from typing import Tuple


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
    def __init__(self, lstm_units, train_backbone = False, imagenet = True):
        super().__init__()
        if not imagenet:
            print('backbone from stratch')
        resnet = models.resnet50(pretrained=imagenet)
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

class CorrLSTM(nn.Module):
    def __init__(self, lstm_units, train_backbone = False, imagenet = True):
        super().__init__()
        if not imagenet:
            print('backbone from stratch')
        resnet = models.resnet50(pretrained=imagenet)
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.train_backbone = train_backbone
        self.backbone = resnet

        input_dim = 2048
        self.lstm = nn.LSTM(input_dim, lstm_units)
        self.fc = nn.Linear(lstm_units, input_dim)

    def train(self, mode=True):
        self.lstm.train(mode)
        self.fc.train(mode)
        self.backbone.train(self.train_backbone and mode)
        if not self.train_backbone and mode:
            print('Fixing resnet in evaluation mode')

    def forward(self, frames):
        self.lstm.flatten_parameters()
        features = feature_extraction(self.backbone, frames)
        F, N, _ = features.size()
        T = F - 1
        features = features.narrow(0, 0, T)
        out_features, _ = self.lstm(features.narrow(0, 0, T - 1))
        out_features = self.fc(out_features.view((T - 1) * N, -1)).view(T - 1, N, -1)
        return out_features, features


class FramePredict(nn.Module):
    def __init__(self, lstm_units, task, train_backbone = False, imagenet = True):
        super().__init__()
        if not imagenet:
            print('Network.Imanget: training from scratch')
        resnet = models.resnet50(pretrained=imagenet)
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

class FrameCorrelationPredict(nn.Module):

    def __init__(self, lstm_units, train_backbone = False):
        """ Construct predictor similar to FramePridict using CorrelationLSTM

        Args:
            lstm_units (int): Number of LSTM units
            train_backbone (bool, optional): Whether to train Resnet or not. Defaults to False.
        """
        super().__init__()
        # Construct Resnet
        resnet = models.resnet50(pretrained=True)
        # Set train Resnet or not
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.train_backbone = train_backbone
        self.backbone = resnet

        # Construct LSTM
        input_dim = 2048
        self.lstm_units = lstm_units
        self.lstm = nn.LSTM(input_dim, lstm_units)
        # Construct FC, input dimension (4 * lstm_unit)
        self.fc = nn.Sequential(
                    nn.Linear((3 * lstm_units), 100),
                    nn.ReLU(),
                    nn.Linear(100, 2)
                    )
        self.register_buffer('pos', torch.LongTensor([1]))
        self.register_buffer('neg', torch.LongTensor([0]))


    def train(self, mode=True):
        for layer in self.children():
            if layer == self.backbone:
                layer.train(self.train_backbone and mode)
                if not self.train_backbone and mode:
                    print('Fixing resnet in evaluation mode')
            else:
                layer.train(mode)

    def forward(self, frames: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """ Build Correlation LSTM for order prediction task
            Predict the order of F - 2 pair of frames (F - 1 frames used)
            Feed 2 frames back and forth for symmetry
            Return predictions and labels in [T * N * 2 * 2]

        Args:
            frames (torch.tensor): Input frames [F * N * C * H * W]
        """
        # Get correlation features [F * N * 2048]
        features = one_kernel_feature_extraction(self.backbone, frames)
        F, N, _ = features.size()
        # Note that our last frame feature is inaccurate due to looping
        # The idea is, the output of consecutive frames and skipping one frame should be different
        T = F - 2
        self.lstm.flatten_parameters()
        # Feed in [0, T] as context
        # I hate to do this but here we loop
        # h, c = torch.zeros((1, N, self.lstm_units)), torch.zeros((1, N, self.lstm_units))
        first, second = [], []
        for t in range(T):
            if t == 0:
                first_output, (h_next, c_next) = self.lstm(features.narrow(0, t, 1))
                second_output, _ = self.lstm(features.narrow(0, t, 1))
            else:
                first_output, (h_next, c_next) = self.lstm(features.narrow(0, t, 1), (h, c))
                second_output, _ = self.lstm(features.narrow(0, t, 1), (h, c))
            first.append(first_output)
            second.append(second_output)
            h, c = h_next, c_next

        # Each one is [T * N * hidden_size]
        last_features = [torch.stack(first), torch.stack(second)]
        # Basic feature engineering, key notion is to choose asymmetrical operations
        # Engineer result is [T * N * (4 * lstm_unit)]
        # Construct features [first, second] and [second first], we'll set labels accordingly
        engineered_features = []
        for i in range(2):
            j = 1 - i
            # print(torch.cross(last_features[i], last_features[j]).size())
            engineer = torch.cat([  last_features[i],
                                    last_features[j],
                                    last_features[i] - last_features[j]], dim = -1)
            engineered_features.append(engineer)
        # Final features [T * (2 * N) * (3 * lstm_units)], can be seen as augmenting input
        final_features = torch.cat(engineered_features, dim = 1)
        predictions = self.fc(final_features.view(T*2*N, -1)).view(T, 2*N, 2)
        # Prepare labels
        labels = torch.cat([self.pos.repeat(N), self.neg.repeat(N)])  # [(2 * N)]
        labels = labels.repeat(T, 1) # [T * (2 * N)]
        return predictions, labels

class FrameCorrelationFeaturePredict(nn.Module):

    def __init__(self, lstm_units, train_backbone = False):
        """ Construct predictor similar to FramePridict using CorrelationLSTM

        Args:
            lstm_units (int): Number of LSTM units
            train_backbone (bool, optional): Whether to train Resnet or not. Defaults to False.
        """
        super().__init__()
        # Construct Resnet
        resnet = models.resnet50(pretrained=True)
        # Set train Resnet or not
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.train_backbone = train_backbone
        self.backbone = resnet

        # Construct LSTM
        input_dim = 2048
        self.lstm_units = lstm_units
        self.lstm = nn.LSTM(input_dim, lstm_units)
        # Construct FC, input dimension (4 * lstm_unit)
        self.compare_fc = nn.Sequential(
            nn.Linear(input_dim * 2, lstm_units),
            nn.ReLU())
        self.predict_mlp = nn.Sequential(
            nn.Linear(lstm_units * 2, 100),
            nn.ReLU(),
            nn.Linear(100, 2)
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

    def forward(self, frames: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """ Build Correlation LSTM for order prediction task
            Predict the order of F - 2 pair of frames (F - 1 frames used)
            Feed 2 frames back and forth for symmetry
            Return predictions and labels in [T * N * 2 * 2]

        Args:
            frames (torch.tensor): Input frames [F * N * C * H * W]
        """
        # Get correlation features [F * N * 2048]
        features = one_kernel_feature_extraction(self.backbone, frames)
        F, N, _ = features.size()
        # Note that our last frame feature is inaccurate due to looping
        # The idea is, the output of consecutive frames and skipping one frame should be different
        T = F - 4
        self.lstm.flatten_parameters()
        lstm_features, _ = self.lstm(features.narrow(0, 0, T))

        features_p = torch.cat([features.narrow(0, 2, T), features.narrow(0, 3, T)], dim = -1)
        features_n = torch.cat([features.narrow(0, 3, T), features.narrow(0, 2, T)], dim = -1)

        concat_features = torch.cat([features_p, features_n], dim = -1).view(T * N * 2, -1)

        lstm_features = lstm_features.repeat(1, 1, 2).view(T * N * 2, -1)
        fuse = torch.cat([lstm_features, self.compare_fc(concat_features)], dim = -1)
        predictions = self.predict_mlp(fuse).view(T, N * 2, 2)

        d = frames.device
        labels = self.label.repeat(T, N, 1).view(T, N * 2)
        return predictions, labels


def l2loss(x, y):
    return torch.sum((x - y) ** 2, dim=-1)


def build_dataset(split, shuffle, config):
    length = config['data']['length']
    t = transforms.Compose([
       transforms.CenterCrop((224, 224)),
       transforms.ToTensor()
       ])
    if split.startswith('train') and config['data'].get('augmentation', 'val') == 'normal':
        print('apply normal transformation on split {}'.format(split))
        t = transforms.Compose([
            transforms.RandomResizedCrop((224, 224), (0.5, 1.0), ratio = (3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(),
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

class CorrSeqPredFramework(SeqPredFramework):
    def build_network(self):
        self.model = FrameCorrelationFeaturePredict(**self.config['network'])

class CorrFramePredFramework(SeqFramework):
    def build_network(self):
        self.model = CorrLSTM(**self.config['network'])

