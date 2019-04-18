import argparse
import os

import numpy as np
import torch
from torch import nn
from torchvision import models

from lib.framework import Framework
from lib.video_data import dataset, sampler, transforms


class ResnetFramework(Framework):
    def build_network(self):
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential()
        self.model = resnet

    def build_dataset(self):
        pass

    def build_optimizer(self):
        pass

    def predict_batch(self, frames, labels):
        frames = torch.cat(frames, dim=0)
        return self.model(frames)

    @staticmethod
    def write_feature(dataset, vid, features):
        video_root = os.path.join(dataset.data_root, os.path.dirname(dataset.videos[vid][0]))
        features = np.concatenate(features, axis=0)
        if features.shape[0] == len(dataset.videos[vid]):
            np.save(os.path.join(video_root, 'resnet_center'), features)
        else:
            print('ERR', video_root, features.shape[0], len(dataset.videos[vid]))

    def save_feature(self, dataset, dataloader):
        self.model.eval()
        vid = None
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                args, indices = self.prepare_data(data)
                result = self.predict_batch(*args)
                result = result.cpu().numpy()
                bound = 0
                for i, v in enumerate(indices.tolist()):
                    if v != vid:
                        if v % self.config['print_freq'] == 0:
                            print('Video', v)
                        if bound < i:
                            features.append(result[bound:i])
                        bound = i
                        if vid is not None:
                            self.write_feature(dataset, vid, features)
                        vid = v
                        features = []
                features.append(result[bound:])
            self.write_feature(dataset, vid, features)
        return metrics

def build_dataset(split, config):
    length = 1
    stride = 1
    data = dataset.build_kinetics_dataset(
            split,
            transforms=transforms.Compose([
               transforms.CenterCrop((224, 224)),
               transforms.ToTensor()
               ]),
            length = 1,
            stride = 1,
            )
    loader = torch.utils.data.DataLoader(
            data,
            sampler=sampler.ValSampler(data,
                stride=1),
            batch_size=config['batch_size'],
            num_workers=config['num_worker']
            )
    return data, loader

def main(config):
    framework = ResnetFramework(config)
    framework.cuda()
    data, loader = build_dataset('train_v1', config)
    framework.save_feature(data, loader)


parser = argparse.ArgumentParser()
parser.add_argument('-p', '--print_freq', type=int, default=100)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-t', '--num_worker', type=int, default=4)
args = parser.parse_args()

config = dict()
config['print_freq'] = args.print_freq
config['batch_size'] = args.batch_size
config['num_worker'] = args.num_worker

main(config)
