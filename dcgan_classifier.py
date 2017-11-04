from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

## Import GANs ##
from GANs import *

## Import Classifiers ##
from classifiers import *
from utils import progress_bar, init_params, weights_init


opt = get_params()
print(opt)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

## Data Loaders ##
source_train_loader, source_test_loader = get_dataset(dataset=opt.sourceDataset, root_dir=opt.sourceroot,
                          imageSize=opt.imageSize, batchSize=opt.batchSize, workers=opt.workers)

target_train_loader, target_test_loader = get_dataset(dataset=opt.targetDataset, root_dir=opt.targetroot,
                          imageSize=opt.imageSize, batchSize=opt.batchSize, workers=opt.workers)

nz = int(opt.nz)
imageSize = int(opt.imageSize)
nc = 1 if opt.targetDataset in ['mnist', 'usps'] else 3

##### Generator #####
# netG = dcgan_G(ngpu, imageSize*imageSize + nz)
netG = dcgan_G(in_channels=nz, out_channels=nc, ngf=opt.ngf, ngpu=opt.ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

##### Discrminator #####
netD = dcgan_D(in_channels=nc, ndf=opt.ndf, ngpu=opt.ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

##### Classifier #####
if opt.netT != '':
    chk = torch.load(opt.netT)
    netT = chk['netT']
    best_acc = chk['acc']
    netT_epoch = chk['epoch']
else:
    netT = ResNet18()
    init_params(netT)
    best_acc = 0
    netT_epoch = 0
print(netT)

criterion = nn.BCELoss()
criterion_T = nn.CrossEntropyLoss()

inputs = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz).uniform_(-1, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netT.cuda()
    criterion.cuda()
    criterion_T.cuda()
    inputs, label = inputs.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_gan, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_gan, betas=(opt.beta1, 0.999))
optimizerT = optim.SGD(netT.parameters(), lr=opt.lr_clf, momentum=0.9, weight_decay=5e-4)


def test(epoch, test_loader, save=True):
    global best_acc
    epoch += netT_epoch + 1
    netT.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if opt.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = netT(inputs)

        loss = criterion_T(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        # print(targets.size(0))
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    # Save checkpoint.
    acc = 100.*correct/total
    if save and (acc > best_acc):
        print('Saving..')
        print("Epoch:", epoch, "Accuracy:", acc)
        state = {
            'net': netT, #.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        torch.save(state, '%s/netT_epoch_%d.pth' %(opt.chkpt, epoch))
        best_acc = acc

# print("Testing on MNIST dataset")
# test(-1, source_train_loader, save=False)
# print("Testing on USPS dataset")
# test(-1, target_test_loader)

target_data_iter = iter(target_train_loader)

for epoch in range(opt.niter):
    netT.train()
    netD.train()
    netG.train()

    for i, source_data in enumerate(source_train_loader, 0):

        # Idefinitely looping over target data loader #
        try:
            target_data = target_data_iter.next()
        except StopIteration:
            target_data_iter = iter(target_train_loader)
            target_data = target_data_iter.next()

        ### Discrminator Step ###
        ################################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z))) #
        ################################################################
        netD.zero_grad()

        ## Target Batch ##
        target_cpu, _ = target_data
        batch_size = target_cpu.size(0)
        if opt.cuda:
            target_cpu = target_cpu.cuda()

        ## Train with target data ##
        # inputs.resize_as_(target_cpu).copy_(target_cpu)
        inputv = Variable(target_cpu)
        output = netD(inputv)
        label.resize_(batch_size).fill_(real_label)
        labelv = Variable(label)
        errD_target = criterion(output, labelv)
        errD_target.backward()
        D_x = output.data.mean()

        ## Source Batch ##
        source_cpu, source_label = source_data
        batch_size = source_cpu.size(0)
        if opt.cuda:
            source_cpu, source_label = source_cpu.cuda(), source_label.cuda()

        ## Train with fake ##
        # Sampling from Uniform Distribution as per the paper #
        noise.resize_(batch_size, nz).uniform_(-1, 1)
        # inputs.resize_as_(source_cpu).copy_(source_cpu)
        # inputs.resize_(batch_size, imageSize*imageSize)
        # noisev = Variable(torch.cat([inputs, noise], 1))
        noisev = Variable(noise)
        fake = netG(noisev)
        output = netD(fake.detach())
        label.resize_(batch_size).fill_(fake_label)
        labelv = Variable(label)
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_target + errD_fake

        ## Update D's params ##
        optimizerD.step()

        #############################
        # (2) Update T network #
        #############################
        netT.zero_grad()

        ## Train with source ##
        inputv = Variable(source_cpu)
        labelv = Variable(source_label)
        output = netT(inputv)
        errT_source = criterion_T(output, labelv)
        errT_source.backward()
        # T_x = output.data.mean()

        ## Train with fake ##
        output = netT(fake.detach())
        errT_fake = criterion_T(output, labelv)     # Same Label as of source images
        errT_fake.backward()
        errT = errT_source + errT_fake

        ## Update T's params ##
        optimizerT.step()

        ### Generator Step ###
        ################################################
        # (3) Update G network: maximize log(D(G(z))) #
        ################################################
        netG.zero_grad()

        ## Generator loss due to discriminator ##
        # Sampling from Uniform Distribution as per the paper #
        noise.resize_(batch_size, nz).uniform_(-1, 1)
        # inputs.resize_as_(source_cpu).copy_(source_cpu)
        # inputs.resize_(batch_size, imageSize*imageSize)
        # noisev = Variable(torch.cat([inputs, noise], 1))
        noisev = Variable(noise)
        fake = netG(noisev)
        output = netD(fake)
        label.resize_(batch_size).fill_(real_label)
        labelv = Variable(label)  # fake labels are real for generator cost
        errG_d = criterion(output, labelv)
        errG_d.backward()
        D_G_z2 = output.data.mean()

        ## Generator loss due to task specific loss ##
        output = netT(fake)
        labelv = Variable(source_label)
        errG_t = criterion_T(output, labelv)     # Same Label as of source images
        errG_t.backward()
        errG = errG_d + errG_t

        ## Update G's params ##
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_T: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(source_train_loader),
                 errD.data[0], errG.data[0], errT.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(target_cpu,
                    '%s/real_samples_target_epoch_%03d.jpeg' % (opt.outf, epoch + netT_epoch + 1),
                    normalize=True)

            vutils.save_image(source_cpu,
                    '%s/real_samples_source_epoch_%03d.jpeg' % (opt.outf, epoch + netT_epoch + 1),
                    normalize=True)

            # Sampling from Uniform Distribution as per the paper
            noise.resize_(batch_size, nz).uniform_(-1, 1)
            # source_cpu, source_label = source_data
            # if opt.cuda:
                # source_cpu, source_label = source_cpu.cuda(), source_label.cuda()
            # source_cpu.resize_(batch_size, ngf*ngf)
            noisev = Variable(torch.cat([inputs, noise], 1))
            # noisev = Variable(noise)
            fake = netG(noisev)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.jpeg' % (opt.outf, epoch + netT_epoch + 1),
                    normalize=True)

    # if epoch % 5 == 0:
        # print("Testing on MNIST dataset")
        # test(epoch, source_train_loader, save=False)
    # print("Testing on USPS dataset")
    # test(epoch, target_test_loader)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.chkpt, epoch + netT_epoch + 1))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.chkpt, epoch + netT_epoch + 1))
print("TRAINING DONE!")
