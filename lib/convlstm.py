import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models
from .base import accuracy
from .framework import Framework


class ConvLSTM(nn.Module):
    def __init__(self, classes, lstm_units):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.backbone = resnet

        self.lstm = nn.LSTM(in_planes, lstm_units)
        self.lstm.flatten_parameters()
        self.fc = nn.Linear(lstm_units, classes)

    def forward(self, frames):
        batch_size = frames[0].size(0)
        steps = len(frames)

        frames = torch.cat(frames, dim=0)
        features = self.backbone(frames).view(steps, batch_size, -1)
        features, _ = self.lstm(features)
        return self.fc(features.view(steps * batch_size, -1))

class CNN(nn.Module):
    def __init__(self, classes):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        in_planes = resnet.fc.in_features
        resnet.fc = nn.Sequential()
        self.backbone = resnet
        self.fc = nn.Linear(in_planes, classes)

    def forward(self, frames):
        batch_size = frames[0].size(0)
        steps = len(frames)

        frames = torch.cat(frames, dim=0)
        features = self.backbone(frames).view(steps, batch_size, -1)
        return self.fc(features.view(steps * batch_size, -1))


class SeqClsMixin:
    def train_batch(self, frames, labels):
        labels = labels.repeat(len(frames))
        logits = self.model(frames)
        loss = nn.CrossEntropyLoss()(logits, labels)
        acc = accuracy(logits, labels)
        return dict(loss=loss, acc=acc)

    def predict_batch(self, frames, labels):
        batch_size = frames[0].size(0)
        steps = len(frames)
        logits = self.model(frames)
        proba = F.softmax(logits, dim=1).view(steps, batch_size, -1)
        proba = torch.mean(proba, dim=0)
        ret = dict(proba=proba)
        return ret

    def eval_predictions(self, dataset, predictions):
        data = np.concatenate([predictions['indices'].reshape(-1, 1),
            predictions['proba']], axis=1)
        proba = pd.DataFrame(data).groupby(0).mean().values
        N = len(dataset.labels)
        loss = -np.mean(np.log(proba[np.arange(0, N), dataset.labels]))
        acc = np.mean(np.argmax(proba, axis=-1) == dataset.labels)
        return dict(loss=loss, acc=acc)

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
