from __future__ import print_function
import argparse
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

from mnistm_loader import *
from usps_loader import *

# Classifier
from models import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--target', required=True, help='mnist-m | usps')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--targetroot', default='.', help='path to target dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
elif opt.dataset == 'mnist':
    dataset = dset.MNIST(root=opt.dataroot, download=True,
                         transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.ToTensor(),
                         ]))
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

if opt.target == 'usps':
    target = USPS(data_dir=opt.targetroot,
                transform=transforms.Compose([
                    transforms.Scale(opt.imageSize),
                    transforms.ToTensor(),
                    ]))

assert target
t_loader = torch.utils.data.DataLoader(target, batch_size=opt.batchSize,
                                        shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 1


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.fc = nn.Sequential(
            # fc layer
            nn.Linear(ngf*ngf+nz, nz),
            nn.ReLU(True)
        )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            temp = nn.parallel.data_parallel(self.fc, input, range(self.ngpu))
            temp = temp.view(input.size(0), -1, 1, 1)
            output = nn.parallel.data_parallel(self.main, temp, range(self.ngpu))
        else:
            temp = self.fc(input).view(input.size(0), -1, 1, 1)
            output = self.main(temp)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

##### Classifier #####
netT = ResNet18()

criterion = nn.BCELoss()
criterion_T = nn.CrossEntropyLoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = source_label = torch.FloatTensor(opt.batchSize)
real_label = target_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netT.cuda()
    criterion.cuda()
    criterion_T.cuda()
    input, label, source_label = input.cuda(), label.cuda(), source_label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerT = optim.SGD(netT.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

for epoch in range(opt.niter):
    for i, (data, t_data) in enumerate(zip(dataloader, t_loader), 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
#        # train with real
#        netD.zero_grad()
#        real_cpu, _ = data
#        batch_size = real_cpu.size(0)
#        if opt.cuda:
#            real_cpu = real_cpu.cuda()
#        input.resize_as_(real_cpu).copy_(real_cpu)
#        label.resize_(batch_size).fill_(real_label)
#        inputv = Variable(input)
#        labelv = Variable(label)
#
#        output = netD(inputv)
#        errD_real = criterion(output, labelv)
#        errD_real.backward()
#        D_x = output.data.mean()

        # train with target data
        netD.zero_grad()
        target_cpu, _ = t_data
        batch_size = target_cpu.size(0)
        if opt.cuda:
            target_cpu = target_cpu.cuda()
        input.resize_as_(target_cpu).copy_(target_cpu)
        label.resize_(batch_size).fill_(target_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_target = criterion(output, labelv)
        errD_target.backward()
        D_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz).normal_(0, 1)
        source_cpu, _ = data
        if opt.cuda:
            source_cpu = source_cpu.cuda()
        source_cpu.resize_(batch_size, ngf*ngf)
        noise = torch.cat([source_cpu, noise], 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_target + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(target_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG_1 = criterion(output, labelv)
        errG_1.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        #############################
        # (3) Update T network
        #############################
        # train with source
        netT.zero_grad()
        source_cpu, source_label = data
        batch_size = source_cpu.size(0)
        if opt.cuda:
            source_cpu, source_label = source_cpu.cuda(), source_label.cuda()
        input.resize_as_(source_cpu).copy_(source_cpu)
        inputv = Variable(input)
        labelv = Variable(source_label)

        output = netT(inputv)
        errT_source = criterion_T(output, labelv)
        errT_source.backward()
        T_x = output.data.mean()

        # train with fake
        noise.resize_(batch_size, nz).normal_(0, 1)
        source_cpu, source_label = data
        if opt.cuda:
            source_cpu, source_label = source_cpu.cuda(), source_label.cuda()
        source_cpu.resize_(batch_size, ngf*ngf)
        noise = torch.cat([source_cpu, noise], 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(source_label)
        output = netT(fake.detach())
        errT_fake = criterion_T(output, labelv)
        errT_fake.backward()
        T_G_z1 = output.data.mean()
        errT = errT_source + errT_fake
        optimizerT.step()

        ############################
        # (4) Update G network
        ############################
        netG.zero_grad()
        labelv = Variable(source_label)  # fake labels are real for generator cost
        output = netT(fake.detach())
        errG_2 = criterion_T(output, labelv)
        errG_2.backward()
        errG = errG_1 + errG_2
        G_z2 = output.data.mean()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_T: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], errT.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(target_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            
            noise.resize_(batch_size, nz).normal_(0, 1)
            source_cpu, source_label = data
            if opt.cuda:
                source_cpu, source_label = source_cpu.cuda(), source_label.cuda()
            source_cpu.resize_(batch_size, ngf*ngf)
            noise = torch.cat([source_cpu, noise], 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
