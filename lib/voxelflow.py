import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torchvision import models
from sync_batchnorm import convert_model
from torch.nn import BatchNorm2d

from .base import accuracy, time_distributed, load_network
from .framework import Framework
from .lstm import SeqPredFramework
from .convlstm import SeqClsMixin

class VoxelFlowFramework(SeqPredFramework):
    def train_batch(self, features, labels):
        steps = features.size(0)
        truth, pred = self.model(features)
        loss = nn.MSELoss()(truth, pred)
        # logits = logits.view(-1, logits.size(2))
        # labels = labels.view(-1)
        # loss = nn.CrossEntropyLoss()(logits, labels)
        return dict(loss=loss)

    def build_network(self):
        self.model = VoxelFlow(**self.config['network']) 

class VoxelFlowFrameworkClassification(SeqClsMixin, Framework):
    def build_network(self):
        self.model = VoxelFlow(classification=True, classes=self.classes, **self.config['network'])
        load_network(self.model, self.config['checkpoint'], strict=False)


def meshgrid(height, width):
    x_t = torch.matmul(
        torch.ones(height, 1), torch.linspace(-1.0, 1.0, width).view(1, width))
    y_t = torch.matmul(
        torch.linspace(-1.0, 1.0, height).view(height, 1), torch.ones(1, width))

    grid_x = x_t.view(1, height, width)
    grid_y = y_t.view(1, height, width)
    return grid_x, grid_y


class VoxelFlow(nn.Module):

    def __init__(self, syn_type='inter', classification=False, classes=None):
        super().__init__()
        self.syn_type = syn_type
        self.input_mean = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.input_std = [0.5 * 255, 0.5 * 255, 0.5 * 255]
        self.classification = classification
        self.classes = classes

        # self.syn_type = config.syn_type

        # bn_param = config.bn_param
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(
            6, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv1_bn = BatchNorm2d(64)

        self.conv2 = nn.Conv2d(
            64, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv2_bn = BatchNorm2d(128)

        self.conv3 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_bn = BatchNorm2d(256)

        self.bottleneck = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bottleneck_bn = BatchNorm2d(256)

        self.deconv1 = nn.Conv2d(
            512, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv1_bn = BatchNorm2d(256)

        self.deconv2 = nn.Conv2d(
            384, 128, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv2_bn = BatchNorm2d(128)

        self.deconv3 = nn.Conv2d(
            192, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.deconv3_bn = BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 3, kernel_size=5, stride=1, padding=2)

        if (classification):
            self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
            self.linear = nn.Linear(256, classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, frames, syn_type='inter'):
        if (syn_type == 'inter'):
            frame1 = frames[0]
            frame2 = frames[2]
            truth = frames[1]
        else:
            frame1 = frames[0]
            frame2 = frames[1]
            truth = frames[2]
        x = torch.cat((frame1, frame2), dim=1)
        input = x
        # input_size = tuple(x.size()[2:4])
        input_size = tuple(frames.size()[3:5])

        x = self.conv1(x)
        x = self.conv1_bn(x)
        conv1 = self.relu(x)

        x = self.pool(conv1)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        conv2 = self.relu(x)

        x = self.pool(conv2)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        conv3 = self.relu(x)

        x = self.pool(conv3)

        x = self.bottleneck(x)
        x = self.bottleneck_bn(x)
        x = self.relu(x)

        if (self.classification):
            # return here
            x = self.avg_pool(x).view(x.size(0), -1)
            out_class = torch.unsqueeze(self.linear(x), 0)
            return out_class

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = self.deconv1_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = self.deconv2_bn(x)
        x = self.relu(x)

        x = nn.functional.upsample(
            x, scale_factor=2, mode='bilinear', align_corners=False)

        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = self.deconv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = nn.functional.tanh(x)

        flow = x[:, 0:2, :, :]
        mask = x[:, 2:3, :, :]

        grid_x, grid_y = meshgrid(input_size[0], input_size[1])
        with torch.cuda.device(input.get_device()):
            grid_x = torch.autograd.Variable(
                grid_x.repeat([input.size()[0], 1, 1])).cuda()
            grid_y = torch.autograd.Variable(
                grid_y.repeat([input.size()[0], 1, 1])).cuda()

        flow = 0.5 * flow

        if self.syn_type == 'inter':
            coor_x_1 = grid_x - flow[:, 0, :, :]
            coor_y_1 = grid_y - flow[:, 1, :, :]
            coor_x_2 = grid_x + flow[:, 0, :, :]
            coor_y_2 = grid_y + flow[:, 1, :, :]
        elif self.syn_type == 'extra':
            coor_x_1 = grid_x - flow[:, 0, :, :] * 2
            coor_y_1 = grid_y - flow[:, 1, :, :] * 2
            coor_x_2 = grid_x - flow[:, 0, :, :]
            coor_y_2 = grid_y - flow[:, 1, :, :]
        else:
            raise ValueError('Unknown syn_type ' + self.syn_type)

        output_1 = torch.nn.functional.grid_sample(
            input[:, 0:3, :, :],
            torch.stack([coor_x_1, coor_y_1], dim=3),
            padding_mode='border')
        output_2 = torch.nn.functional.grid_sample(
            input[:, 3:6, :, :],
            torch.stack([coor_x_2, coor_y_2], dim=3),
            padding_mode='border')

        mask = 0.5 * (1.0 + mask)
        mask = mask.repeat([1, 3, 1, 1])
        x = mask * output_1 + (1.0 - mask) * output_2

        return truth, x