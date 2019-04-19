import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models
from sync_batchnorm import convert_model

from .base import accuracy, time_distributed, load_network
from .framework import Framework


class ConvLSTM(nn.Module):
    def __init__(self, classes, lstm_units, sync_bn=False, load_lstm=None):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        if sync_bn:
            print('Convert model using sync bn')
            resnet = convert_model(resnet)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
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
        steps = frames.size(0)
        batch_size = frames.size(1)
        labels = labels.repeat(steps) # T * B
        logits = self.model(frames).view(steps * batch_size, -1)
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels)
        return dict(loss=loss, acc=acc)

    def predict_batch(self, frames, labels):
        steps = frames.size(0)
        batch_size = frames.size(1)
        logits = self.model(frames)
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
