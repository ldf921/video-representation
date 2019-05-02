import argparse
import os
import sys
sys.path.append('/app')

import torch
import yaml

from lib.convlstm import ConvLSTMFramework, CNNFramework
from lib.voxelflow import VoxelFlowFramework, VoxelFlowFrameworkClassification
from lib.base import load_network
from lib.lstm import SeqFramework, SeqPredFramework
from lib.stackcnn import stackcl_framework


parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--test', type=lambda x : x.split(','), default=None)
parser.add_argument('-s', '--suffix', type=str, default=None)
parser.add_argument('-c', '--checkpoint', type=str, default=None)
parser.add_argument('-p', '--print_freq', type=int, default=100)
parser.add_argument('-n', '--num_epochs', type=int, default=20)
parser.add_argument('-b', '--batch_size', type=int, default=8)
parser.add_argument('-t', '--num_worker', type=int, default=4)

args = parser.parse_args()
with open(args.config + '.yaml', 'r') as fi:
    config = yaml.load(fi)

config['print_freq'] = args.print_freq
config['batch_size'] = args.batch_size
config['num_worker'] = args.num_worker

if args.suffix is not None:
    for arg in args.suffix.split('_'):
        if arg.startswith('lr'):
            config['optimizer']['lr'] = float(arg[2:])
            print('Setting lr', config['optimizer']['lr'])
        elif arg.startswith('wd'):
            config['optimizer']['weight_decay'] = float(arg[2:])
            print('Setting wd', config['optimizer']['weight_decay'])
        elif arg == 'moment':
            config['optimizer']['type'] = 'SGD'
            config['optimizer']['momentum'] = 0.9
        else:
            print('Suffix', arg)

if config['framework'] == 'convlstm':
    framework = ConvLSTMFramework(config)
elif config['framework'] == 'cnn':
    framework = CNNFramework(config)
elif config['framework'] == 'seq':
    framework = SeqFramework(config)
elif config['framework'] == 'seqpred':
    framework = SeqPredFramework(config)
elif config['framework'].startswith('stackcl'):
    framework = stackcl_framework(config)
elif config['framework'] == 'voxelflow':
    framework = VoxelFlowFramework(config)
elif config['framework'] == 'voxelflow_class':
    framework = VoxelFlowFrameworkClassification(config)
    
CP_ROOT = 'checkpoints'
if args.test is not None:
    exp_dir = os.path.dirname(args.config)
else:
    exp_dir = os.path.join(CP_ROOT, args.config.replace('config/', ''))
    if args.suffix is not None:
        exp_dir += "_" + args.suffix
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, 'config.yaml'), 'w') as fo:
        yaml.dump(config, fo)

def save(framework, epoch):
    torch.save(framework.model.state_dict(), os.path.join(exp_dir, '%d.model' % epoch))

def print_dict(metrics, prefix):
    msg = "{} ".format(prefix)
    for k, v in metrics.items():
        msg += '{}: {:.4f}, \t'.format(k, v)
    return msg

def train(framework):
    framework.cuda()
    with open(os.path.join(exp_dir, 'log'), 'a') as flog:
        for epoch in range(1, args.num_epochs + 1):
            train_result = framework.train_epoch(epoch)
            save(framework, epoch)
            val_result = framework.valid()
            msg = 'Epoch [{}/{}], {}{}'.format(epoch, args.num_epochs,
                    print_dict(train_result, 'Train'), print_dict(val_result, 'Val'))
            print(msg)
            print(msg, file=flog)
            flog.flush()

def generator_k(k, loader):
    for i, data in enumerate(loader):
        if i > k:
            break
        yield data

def test(framework):
    from utils import saveimg
    load_network(framework.model, os.path.join(exp_dir, args.checkpoint))
    framework.cuda()
    if config['framework'] == 'voxelflow':
        results = framework.predict(generator_k(1, framework.val_loader))
        saveimg(results['truth'], results['pred'], os.path.join(exp_dir, 'vis.jpg'), n=10)
        print(((resutls['truth'] - results['pred']) ** 2).mean())
    for subset in args.test:
        data, loader = framework.build_test_dataset(subset)
        result = framework.test(data, loader)
        print(print_dict(result, subset))

if args.test is not None:
    test(framework)
else:
    train(framework)
