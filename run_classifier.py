from __future__ import print_function
import os, sys
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torch.autograd import Variable

from utils import progress_bar, init_params
from dataset import get_dataset
from classifiers import *
from plotter import Plotter

parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--dataset', required=True, help='mnist | mnistm | usps')
parser.add_argument('--datadir', required=True, help='path to dataset')
parser.add_argument('--batchsize', type=int, default=32, help='training batch size')
parser.add_argument('--imagesize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--chkpt', default='checkpoint', help='folder to save model checkpoints')
parser.add_argument('--plotdir', default='plots', help='path to save plots')
parser.add_argument('--net', default='', help="path to net (to continue training)")
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--lr_decay_rate', type=float, default=0.95, help='learning rate decay rate')
parser.add_argument('--lr_decay_step', type=int, default=20000, help='learning rate decay step')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=9926, help='manual seed')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 weight decay')

args = parser.parse_args()

random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

if args.cuda and torch.cuda.is_available():
    use_cuda = True
    torch.cuda.manual_seed_all(args.manualSeed)
else:
    use_cuda = False
best_acc = 0  # best test accuracy
best_epoch = 0  # epoch at which test accuracy is best
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainloader, testloader = get_dataset(dataset=args.dataset, root_dir=args.datadir,
                          imageSize=args.imagesize, batchSize=args.batchsize, workers=args.workers)

num_channels = 1 if args.dataset in ['mnist', 'usps'] else 3
num_classes = 10
ngpu = args.ngpu

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.net), 'Error: no saved model found!'
    checkpoint = torch.load(args.net)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('==> Building model..')
    classifier_name = "MnistClassifier"
    net = MnistClassifier(1, num_channels, num_classes, ngpu)
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
print(net)

if use_cuda:
    net.cuda()
    cudnn.benchmark = True
    if ngpu >1:
        net = torch.nn.DataParallel(net, device_ids=range(ngpu))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)
net_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

if not os.path.isdir(args.plotdir):
    os.makedirs(args.plotdir)
plot_loss = Plotter("%s/%s_%s_loss.jpeg" % (args.plotdir, classifier_name, args.dataset), num_lines=2,
                    legends=["train_loss", "test_loss"], xlabel="Number of Epochs", ylabel="Loss",
                    title="Loss vs Epochs")

plot_acc = Plotter("%s/%s_%s_acc.jpeg" % (args.plotdir, classifier_name, args.dataset), num_lines=2,
                    legends=["train_accuracy", "test_accuracy"], xlabel="Number of Epochs", ylabel="Accuracy",
                    title="Accuracy vs Epochs")
plotters = [plot_loss, plot_acc]

# Training
def train(epoch):
    print('Epoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net_lr_scheduler.step()
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/len(trainloader), 100.*correct/total

def test(epoch):
    global best_acc, best_epoch
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    embeddings = []
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)

        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
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
        if not os.path.isdir(args.chkpt):
            os.makedirs(args.chkpt)
        torch.save(state, '%s/%s_%s.chkpt' % (args.chkpt, classifier_name, args.dataset))
        best_acc = acc
        best_epoch = epoch
    return test_loss/len(testloader), acc

for epoch in range(start_epoch, start_epoch+args.nepoch):
    print("Train:")
    train_loss, train_acc = train(epoch)
    print("Test:")
    test_loss, test_acc = test(epoch)
    plot_loss((epoch, train_loss), (epoch, test_loss))
    plot_acc((epoch, train_acc), (epoch, test_acc))
print("Best accuracy: ", best_acc, "Epoch:", best_epoch)
map(lambda plots: plots.queue.join(), plotters)
map(lambda plots: plots.clean_up(), plotters)
