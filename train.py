import argparse
import os
import sys
sys.path.append('/app')

import torch
import yaml

from lib.convlstm import ConvLSTMFramework, CNNFramework
from lib.base import load_network
from lib.lstm import SeqFramework


parser = argparse.ArgumentParser()
parser.add_argument('config')
parser.add_argument('--test', action='store_true')
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

if config['framework'] == 'convlstm':
    framework = ConvLSTMFramework(config)
elif config['framework'] == 'cnn':
    framework = CNNFramework(config)
elif config['framework'] == 'seq':
    framework = SeqFramework(config)

CP_ROOT = 'checkpoints'
if args.test:
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
            val_result = framework.valid()
            msg = 'Epoch [{}/{}], {}{}'.format(epoch, args.num_epochs,
                    print_dict(train_result, 'Train'), print_dict(val_result, 'Val'))
            save(framework, epoch)
            print(msg)
            print(msg, file=flog)

def test(framework):
    load_network(framework.model, os.path.join(exp_dir, args.checkpoint))
    framework.cuda()
    data, loader = framework.build_test_dataset('test1')
    result = framework.test(data, loader)
    print(print_dict(result, 'Test'))

if args.test:
    test(framework)
else:
    train(framework)
