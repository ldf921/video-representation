import os
import pickle as pkl

import torch
import numpy as np
from PIL import Image

class VideoList:
    def __init__(self, root, frames):
        self.root = root
        self.frames = frames

    def __len__(self):
        return self.frames

    def __getitem__(self, idx):
        return self.root + '/%06d.jpg' % (idx + 1)

class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, videos, labels, data_root, transforms, length, stride):
        self.videos = videos
        self.labels = labels
        self.data_root = data_root
        self.length = length
        self.stride = stride
        self.transforms = transforms

    def open_image(self, img_path):
        return Image.open(os.path.join(self.data_root, img_path))

    def __getitem__(self, idx):
        vid, s = idx

        frames = []
        N = len(self.videos[vid])
        for i in range(s, s + self.length * self.stride, self.stride):
            frames.append(self.open_image(self.videos[vid][i % N]))
        frames = self.transforms(frames)

        labels = self.labels[vid]

        return frames, labels, vid

    def __len__(self):
        return len(self.videos)


class VideoFeatureDataset(VideoDataset):
    def __getitem__(self, idx):
        vid, s = idx

        frames = []
        N = len(self.videos[vid])
        features = np.load(os.path.join(self.data_root, self.videos[vid].root, 'resnet_center.npy'))
        for i in range(s, s + self.length * self.stride, self.stride):
            frames.append(torch.Tensor(features[i % N]))
        labels = self.labels[vid]

        return frames, labels, vid


def build_ucf101_dataset(split, transforms, length, stride, config):
    data_root = 'data/UCF101'
    with open(os.path.join(data_root, '{}.pkl'.format(split)), 'rb') as fi:
        data = pkl.load(fi)
    if isinstance(data['frames'][0], tuple):
        frames = [VideoList(*d)for d in data['frames'] if d[1] > 0]
    else:
        frames = data['frames']
    dataset = VideoDataset(frames, np.array(data['labels']) - 1, data_root, transforms, length, stride)
    dataset.classes = 101
    print('UCF101-{} Samples: {}'.format(split, len(dataset)))
    return dataset


def build_kinetics_dataset(split, transforms, length, stride, feature=False):
    data_root = '/data/kinetics-400'
    with open(os.path.join(data_root, '{}.pkl'.format(split)), 'rb') as fi:
        data = pkl.load(fi)
    frames = [VideoList(*d)for d in data['frames'] if d[1] > 0]
    if feature:
        dataset = VideoFeatureDataset(frames, data['labels'], data_root, transforms, length, stride)
    else:
        dataset = VideoDataset(frames, data['labels'], data_root, transforms, length, stride)
    dataset.classes = 400
    print('Kinetics400-{} Samples: {}'.format(split, len(dataset)))
    return dataset
