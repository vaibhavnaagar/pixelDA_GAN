'''
DCGAN in PyTorch:
https://github.com/pytorch/examples/tree/master/dcgan
Reference:
[1] Radford, Alec, Luke Metz, and Soumith Chintala.
 "Unsupervised representation learning with deep convolutional generative adversarial networks."
  arXiv preprint arXiv:1511.06434 (2015).
'''

import torch
import torch.nn as nn


class dcgan_G(nn.Module):
    def __init__(self, in_channels, out_channels, ngf=64, ngpu=1):
        super(dcgan_G, self).__init__()
        self.ngpu = ngpu
        # self.fc = nn.Sequential(
        #     # fc layer
        #     nn.Linear(ngf*ngf+nz, nz),
        #     nn.ReLU(True)
        # )
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            # temp = nn.parallel.data_parallel(self.fc, inputs, range(self.ngpu))
            inputs = inputs.view(inputs.size(0), -1, 1, 1)
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            # inputs = self.fc(inputs).view(inputs.size(0), -1, 1, 1)
            inputs = inputs.view(inputs.size(0), -1, 1, 1)
            output = self.main(inputs)
        return output

class dcgan_D(nn.Module):
    def __init__(self, in_channels, ndf=64, ngpu=1):
        super(dcgan_D, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)

        return output.view(-1, 1).squeeze(1)
