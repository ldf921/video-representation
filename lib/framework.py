import time
from copy import copy
from collections import defaultdict

import torch
import numpy as np
from torch import nn
from sync_batchnorm import DataParallelWithCallback

from .video_data import dataset, sampler, transforms
from .base import accuracy


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Framework:
    batch_axis = 1

    def __init__(self, config):
        self.config = config
        self.build_dataset()
        self.build_network()
        self.build_optimizer()

    def get_param(self, keys, default=None):
        node = self.config
        for key in keys.split('.'):
            if key in node:
                node = node[key]
            else:
                return default
        return node

    def cuda(self):
        self.model.cuda()
        if self.get_param('network.sync_bn', False):
            self.model = DataParallelWithCallback(self.model, dim=self.batch_axis)
        else:
            self.model = nn.DataParallel(self.model, dim=self.batch_axis)

    def build_optimizer(self):
        args = copy(self.config['optimizer'])
        if args['type'] == 'SGD':
            optim_class = torch.optim.SGD
        elif args['type'] == 'Adam':
            optim_class = torch.optim.Adam
        args.pop('type')
        if isinstance(args['lr'], list):
            args['lr'] = args['lr'][0][0]
        self.optimizer = optim_class(self.model.parameters(), **args)

    def set_learning_rate(self, epoch):
        lrs = self.config['optimizer']['lr']
        if isinstance(lrs, list):
            c = 0
            for lr, num_epochs in lrs:
                c += num_epochs
                if epoch <= c:
                    break
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            print('setting learning rate {}'.format(lr))


    def build_dataset(self):
        length = self.config['data']['length']
        stride = 5
        self.train_data = dataset.build_ucf101_dataset(
                'traindev1',
                transforms=transforms.Compose([
                    transforms.RandomResizedCrop((224, 224), (0.5, 1.0), ratio = (3 / 4, 4 / 3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor()
                    ]),
                length = length,
                stride = stride,
                config = self.config
                )
        self.train_loader = torch.utils.data.DataLoader(
                self.train_data,
                sampler=sampler.RandomSampler(self.train_data, loop=self.config['data']['loop']),
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_worker']
                )
        self.val_data, self.val_loader = self.build_test_dataset('val1')
        self.classes = self.train_data.classes

    def build_test_dataset(self, split):
        length = self.config['data']['length']
        stride = 5
        data = dataset.build_ucf101_dataset(
                split,
                transforms=transforms.Compose([
                   transforms.CenterCrop((224, 224)),
                   transforms.ToTensor()
                   ]),
                length = length,
                stride = stride,
                config = self.config
                )
        loader = torch.utils.data.DataLoader(
                data,
                sampler=sampler.ValSampler(data,
                    stride=length),
                batch_size=self.config['batch_size'],
                num_workers=self.config['num_worker']
                )
        return data, loader

    # Why Non_Blocking

    def prepare_data(self, data):
        frames = torch.stack(data[0], dim=0).cuda(non_blocking=True)
        labels = data[1].cuda(non_blocking=True)
        vids = data[2].numpy()
        return (frames, labels), vids


    def train_epoch(self, epoch):
        self.model.train()
        self.set_learning_rate(epoch)
        end = time.time()
        metrics = defaultdict(AverageMeter)
        for i, data in enumerate(self.train_loader):
            # measure data loading time
            metrics['data_time'].update(time.time() - end)

            args, _ = self.prepare_data(data)
            batch_size = args[0].size(1)
            result = self.train_batch(*args)
            loss = result['loss']
            for k, v in result.items():
                metrics[k].update(v.item(), batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # measure elapsed time
            metrics['batch_time'].update(time.time() - end)
            end = time.time()

            if i % self.config['print_freq'] == 0:
                print('Epoch: [{0}][{1}/{2}]\t'.format(
                       epoch, i, len(self.train_loader)), end='')

                for k, v in metrics.items():
                    print('{key} {val.avg:.3f}'.format(key=k, val=v), end='\t')
                print()
            # if i > 200:
            #     break

        metrics.pop('batch_time')
        metrics.pop('data_time')
        return {k : v.avg for k, v in metrics.items()}
        


    def predict(self, dataloader):
        self.model.eval()
        metrics = defaultdict(list)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                args, indices = self.prepare_data(data)
                result = self.predict_batch(*args)
                for k, v in result.items():
                    metrics[k].append(v.cpu().numpy())
                metrics['indices'].append(indices)
                if i % self.config['print_freq'] == 0:
                    print('Valid {}/{}'.format(i, len(dataloader)))
        for k in metrics:
            metrics[k] = np.concatenate(metrics[k], axis=0)
        return metrics

    def evaluate(self, dataloader):
        self.model.eval()
        metrics = defaultdict(AverageMeter)
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                args, indices = self.prepare_data(data)
                batch_size = args[0][0].size(0)
                result = self.eval_batch(*args)
                for k, v in result.items():
                    metrics[k].update(v.item(), batch_size)
                if i % self.config['print_freq'] == 0:
                    print('Valid {}/{}'.format(i, len(dataloader)))
        return {k : v.avg for k, v in metrics.items()}


