from __future__ import print_function

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
import json

import distiller
import load_settings

from helper.logger import Logger
from helper.util import get_lr, Dict2Obj

# get config file
parser = argparse.ArgumentParser('config file')
parser.add_argument('--config_file', '-c', type=str, help='config file')
config = parser.parse_args()
print('Config file: ', config.config_file)

# load config
args = {}
with open(config.config_file, 'rt') as f:
    args.update(json.load(f))
args = Dict2Obj(args)
print('   Save folder: ', args.exp_path)
print()

# environment set
print('==> Set Environment..')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

# data
print('==> Load data..')
use_cuda = torch.cuda.is_available()
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])


# transform_train = transforms.Compose([
#     transforms.Pad(4, padding_mode='reflect'),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor(),
#     transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
#                          np.array([63.0, 62.1, 66.7]) / 255.0)
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
#                          np.array([63.0, 62.1, 66.7]) / 255.0),
# ])

trainset = torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testset = torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

# Model
print('==> Load model..')
t_net, s_net = load_settings.load_paper_settings(args, n_cls=100)

# Module for distillation
d_net = distiller.Distiller(t_net, s_net, kd_T=args.kd_T)

print('the number of teacher model parameters: {}'.format(sum([p.data.nelement() for p in t_net.parameters()])))
print('the number of student model parameters: {}'.format(sum([p.data.nelement() for p in s_net.parameters()])))

if use_cuda:
    torch.cuda.set_device(0)
    d_net.cuda()
    s_net.cuda()
    t_net.cuda()
    cudnn.benchmark = True

criterion_CE = nn.CrossEntropyLoss()

# Training
def train_with_distill(d_net, epoch, args):
    epoch_start_time = time.time()
    print('\nDistillation epoch: %d' % epoch)

    d_net.train()
    d_net.s_net.train()
    d_net.t_net.train()

    train_loss = 0
    correct = 0
    total = 0

    global optimizer
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        batch_size = inputs.shape[0]
        outputs, loss_kd, loss_distill = d_net(inputs)
        loss_CE = criterion_CE(outputs, targets)

        loss = loss_CE * args.gamma + loss_kd * args.alpha + loss_distill.sum() / batch_size / args.beta

        loss.backward()
        optimizer.step()

        train_loss += loss_CE.item()

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()

        b_idx = batch_idx

    print('Train \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (b_idx + 1), 100. * correct / total, correct, total))

    return train_loss / (b_idx + 1), 100. * correct / total

def test(net):
    epoch_start_time = time.time()
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion_CE(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float().item()
        b_idx = batch_idx

    print('Test \t Time Taken: %.2f sec' % (time.time() - epoch_start_time))
    print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss / (b_idx + 1), 100. * correct / total, correct, total))
    return test_loss / (b_idx + 1), 100. * correct / total


print('==> Evaluate teacher..')
print('   Performance of teacher network')
test(t_net)

print('==> Training..')
time_start = time.time()
logger = Logger(os.path.join(args.exp_path, 'log.txt'), title='log')
logger.set_names(['Epoch', 'lr', 'Time-elapse(Min)',
                  'Train-Loss', 'Train-Acc',
                  'Test-Loss', 'Test-Acc'])

for epoch in range(args.epochs):
    # if epoch is 0:
    #     optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
    #                           lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # elif epoch is (args.epochs // 2):
    #     optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
    #                           lr=args.lr / 10, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    # elif epoch is (args.epochs * 3 // 4):
    #     optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
    #                           lr=args.lr / 100, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    if epoch is 0:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is 150: 
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 10, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is 180:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 100, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
    elif epoch is 210:
        optimizer = optim.SGD([{'params': s_net.parameters()}, {'params': d_net.Connectors.parameters()}],
                              lr=args.lr / 1000, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    train_loss, train_acc = train_with_distill(d_net, epoch, args)
    test_loss, test_acc = test(s_net)

    logs = [epoch+1, get_lr(optimizer), (time.time() - time_start)/60]
    logs += [train_loss, train_acc, test_loss, test_acc]
    logger.append(logs)

logger.close()

print('==> Save last model..')
state_dict = dict(epoch=epoch+1, state_dict=s_net.state_dict(), test_acc=test_acc)
name = os.path.join(args.exp_path, 'student_last.pth')
os.makedirs(os.path.dirname(name), exist_ok=True)
torch.save(state_dict, name)
print('==> Done..')
