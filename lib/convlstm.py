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
    def __init__(self, classes, lstm_units, sync_bn=False, load_lstm=None, load_backbone=None):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
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
        features = time_distributed(self.backbone, frames)
        features, _ = self.lstm(features)
        return time_distributed(self.fc, features)

class CorrelationLSTM(nn.Module):

    def __init__(self, classes, lstm_units, load_lstm=None, load_backbone=None):
        """Construct a CNN-LSTM that computes correlation of last 3 layer of ResNet on consequtive frames.
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
        self.backbone = resnet

        # Construct LSTM
        # Calculate LSTM input dimension 28^2 + 14^2 + 7^2 = 1029
        input_dim = 1029
        self.lstm = nn.LSTM(input_dim, lstm_units)
        if load_lstm is not None:
            load_network(self.lstm, load_lstm, 'module.lstm.')
        self.fc = nn.Linear(lstm_units, classes)

    def feature_extraction(self, model: nn.Module, input: torch.tensor) -> torch.tensor:
        """Extract features from intermediate layers using register_forward_hook
           Return dim [F * N * input_dim]
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
        
    def forward(self, frames):
        self.lstm.flatten_parameters()
        features = feature_extraction(frames)
        features, _ = self.lstm(features)
        F, N, _ = features.size()
        output = self.fc(features.view(F * N, -1))
        return output.view(F, N, -1)

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

class CorrelationLSTMFramework(SeqClsMixin, Framework):
    def build_network(self):
        self.model = CorrelationLSTM(self.classes, **self.config['network'])
