import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models
from sync_batchnorm import convert_model

from .base import*
from .corrnet import CorrNet
from .framework import Framework


class ConvLSTM(nn.Module):
    def __init__(self, classes, lstm_units, pool='avgpool', pretrain=True, sync_bn=False, load_lstm=None, load_backbone=None, train_backbone=True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrain)
        in_planes = resnet.fc.in_features
        if pool == 'groupconv':
            resnet.avgpool = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, (7, 7), groups=32, bias=False)
                )
            print('Using group conv for pooling')
        resnet.fc = nn.Sequential()
        if load_backbone is not None:
            load_network(resnet, load_backbone, 'module.backbone.')
        if sync_bn:
            print('Convert model using sync bn')
            resnet = convert_model(resnet)
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        self.backbone = resnet

        self.lstm = nn.LSTM(in_planes, lstm_units)
        if load_lstm is not None:
            load_network(self.lstm, load_lstm, 'module.lstm.')
        self.fc = nn.Linear(lstm_units, classes)

    def forward(self, frames):
        self.lstm.flatten_parameters()
        features = time_distributed(self.backbone, frames)
        features, _ = self.lstm(features)
        return time_distributed(self.fc, features)

class CorrelationLSTM(nn.Module):

    def __init__(self, classes, lstm_units, load_lstm=None, load_backbone=None, train_backbone=False, load_corr=None):
        """ Construct a CNN-LSTM that computes correlation of last 3 layer of ResNet on consequtive frames.
            [F * N * C * H * W] [F * N * C * H * W] -> [F * N * H * W]
            Each frame is encoded into 28^2 + 14^2 + 7^2 = 1029
        
        Args:
            classes (int): number of classes
            lstm_units (int): output dim of LSTM
            load_lstm ([bool], optional): Load LSTM model. Defaults to None.
            load_backbone ([bool], optional): Load Resnet model. Defaults to None.
        """
        super().__init__()
        # Construct Resnet
        resnet = models.resnet50(pretrained=True)
        if load_backbone is not None:
            load_network(resnet, load_backbone, 'module.backbone.')
        for param in resnet.parameters():
            param.requires_grad = train_backbone
        resnet.fc = nn.Sequential()
        self.backbone = resnet
        self.D = 7
        self.corrnet = CorrNet(self.D**2)
        if load_corr is not None:
            load_network(self.corrnet, load_corr, "module.corrnet.")

        # Construct LSTM
        # input_dim = 512
        input_dim = 2048 + 512
        self.lstm = nn.LSTM(input_dim, lstm_units)
        if load_lstm is not None:
            load_network(self.lstm, load_lstm, 'module.lstm.')
        self.fc = nn.Linear(lstm_units, classes)
        # self.fc = nn.Sequential(nn.Linear(lstm_units, 200),
        #                         nn.ReLU(),
        #                         nn.Linear(200, classes))
        
    def forward(self, frames):
        self.lstm.flatten_parameters()
        feature_map, x, F, N = next_feature_extraction(self.backbone, frames, self.D, (self.D -1)//2)
        features = self.corrnet(feature_map)
        features = torch.cat([features.view(F, N, -1), x.view(F, N, -1)], dim=-1)
        T = F - 1
        features = features.narrow(0, 0, T)
        features, _ = self.lstm(features)
        output = self.fc(features.view(T * N, -1))
        return output.view(T, N, -1)

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.backbone = resnet
        self.fc = nn.Linear(in_planes, classes)

    def forward(self, frames):
        features = time_distributed(self.backbone, frames)
        return time_distributed(self.fc, features)


class SeqClsMixin:
    def train_batch(self, frames, labels):
        logits = self.model(frames)
        steps, batch_size, _ = logits.size()
        logits = logits.view(steps * batch_size, -1)
        labels = labels.repeat(steps) # T * B
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels)
        return dict(loss=loss, acc=acc)

    def predict_batch(self, frames, labels):
        logits = self.model(frames)
        steps, batch_size, _ = logits.size()
        proba = F.softmax(logits, dim=-1)
        proba = torch.mean(proba, dim=0) # B * Classes

        loss = nn.CrossEntropyLoss(reduction='none')(logits.view(steps * batch_size, -1), labels.repeat(steps))
        loss = loss.view(steps, batch_size).mean(dim=0)
        ret = dict(raw_loss=loss, proba=proba)
        return ret

    def eval_predictions(self, dataset, predictions):
        data = np.concatenate([predictions['indices'].reshape(-1, 1),
            predictions['proba']], axis=1)
        proba = pd.DataFrame(data).groupby(0).mean().values
        N = len(dataset.labels)
        loss = -np.mean(np.log(proba[np.arange(0, N), dataset.labels]))
        acc = np.mean(np.argmax(proba, axis=-1) == dataset.labels)

        data = np.stack([predictions['indices'], predictions['raw_loss']], axis=1)
        raw_loss = pd.DataFrame(data).groupby(0).mean().values
        print(raw_loss.shape)
        raw_loss = np.mean(raw_loss)

        return dict(raw_loss=raw_loss, loss=loss, acc=acc)

    def valid(self):
        predictions = self.predict(self.val_loader)
        return self.eval_predictions(self.val_data, predictions)

    def test(self, data, loader):
        predictions = self.predict(loader)
        return self.eval_predictions(data, predictions)


class ConvLSTMFramework(SeqClsMixin, Framework):
    def build_network(self):
        self.model = ConvLSTM(self.classes, **self.config['network'])


class CNNFramework(SeqClsMixin, Framework):
    def build_network(self):
        self.model = CNN(self.classes)

class CorrelationLSTMFramework(SeqClsMixin, Framework):
    def build_network(self):
        self.model = CorrelationLSTM(self.classes, **self.config['network'])
