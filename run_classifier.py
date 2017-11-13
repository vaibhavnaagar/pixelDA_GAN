from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from classifiers import *
from utils import progress_bar, init_params
from torch.autograd import Variable
import pickle
import numpy as np
import sys

from dataset import get_dataset


parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainloader, testloader = get_dataset(dataset="mnist", root_dir="data",
                          imageSize=64, batchSize=32, workers=2)

t_trainloader, t_testloader = get_dataset(dataset="mnistm", root_dir="data",
                          imageSize=64, batchSize=32, workers=2)

source_channels = 1
target_channels = 3
num_classes = 10
ngpu = 1

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7')
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = MnistClassifier(source_channels, target_channels, num_classes, ngpu)
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizerT = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=1e-5)

class NetFeatures(nn.Module):
    def __init__(self, original_model):
        super(NetFeatures, self).__init__()
        # print(list(original_model.children())[:-1])
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        ## For PreActResnet only ##
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        return x

net_features = NetFeatures(net)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs, dataset="source")
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch, get_features=False):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    embeddings = []
    for batch_idx, (inputs, targets) in enumerate(t_testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        if get_features:
            embeddings.append(net_features(inputs).cpu().data.numpy())
            # print(net_features(inputs).cpu().data.numpy().shape)

        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(t_testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        print("Epoch:", epoch, "Accuracy:", acc)
        state = {
            'net': net, #.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = acc
        if get_features:
            embeddings = np.vstack(tuple(embeddings))
            print("Embeddings:", embeddings.shape)
            with open('cifar10_fc.pkl', 'wb') as f:
                pickle.dump(embeddings, f)


for epoch in range(start_epoch, start_epoch+100):
    train(epoch)
    test(epoch, get_features=False)
