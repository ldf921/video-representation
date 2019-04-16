import os
import pickle as pkl

import torch
import numpy as np
from PIL import Image

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

def build_ucf101_dataset(split, transforms, length, stride, config):
    data_root = '/data/UCF101'
    with open(os.path.join(data_root, '{}.pkl'.format(split)), 'rb') as fi:
        data = pkl.load(fi)
    dataset = VideoDataset(data['frames'], np.array(data['labels']) - 1, data_root, transforms, length, stride)
    dataset.classes = 101
    print('UCF101-{} Samples: {}'.format(split, len(dataset)))
    return dataset








