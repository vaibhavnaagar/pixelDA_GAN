'''
GAN used in the paper[1]:
Reference:
[1] Bousmalis, Konstantinos, et al.
    "Unsupervised Pixel-level Domain Adaptation with GANs." (2017)..
'''

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class project_noise(nn.Module):
    def __init__(self, in_features, out_features, ngpu=1):
        super(project_noise, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # batch_size x in_features
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(True)
            # batch_size x out_features
        )

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output


class residual_block(nn.Module):
    def __init__(self, in_channels, filters=64, ngpu=1, kernel_size=3, stride=1, padding=1):
        super(residual_block, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # batch_size x in_channels x 64 x 64
            nn.Conv2d(in_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(True),
            nn.Conv2d(filters, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(filters)
            # batch_size x filters x 64 x 64
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != filters:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filters)
            )

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        output += self.shortcut(inputs)
        return output


class pixelda_G(nn.Module):
    def __init__(self, out_channels, image_size, opt, kernel_size=3, stride=1, padding=1):
        super(pixelda_G, self).__init__()
        assert len(image_size) == 3
        self.ngpu = opt.ngpu
        noise_channels = opt.G_noise_channels
        filters = opt.ngf
        self.noise_size = [noise_channels] + image_size[:2]
        self.noise_layer = project_noise(opt.G_noise_dim, int(np.prod(self.noise_size)), ngpu=opt.ngpu)
        blocks = []
        for block in range(opt.G_residual_blocks):
            blocks.append(residual_block(filters, filters, opt.ngpu, kernel_size, stride, padding))
        self.main = nn.Sequential(
            # batch_size x (image_channels + noise_channels) x H x W
            nn.Conv2d(image_size[-1]+noise_channels, filters, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU(True),
            *blocks,
            nn.Conv2d(filters, out_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Tanh()
            # batch_size x out_channels x H x W
        )

    def forward(self, inputs, noise_vector):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            noise = nn.parallel.data_parallel(self.noise_layer, noise_vector, range(self.ngpu))
            noise_image = noise.view(noise.size(0), *self.noise_size)
            inputs = torch.cat((inputs, noise_image), dim=1)
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            noise = self.noise_layer(noise_vector)
            noise_image = noise.view(noise.size(0), *self.noise_size)
            inputs = torch.cat((inputs, noise_image), dim=1)
            output = self.main(inputs)
        return output


############################
# Discrminator
############################

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, ngpu=1, kernel_size=3, stride=1, padding=1, leakiness=0.2):
        super(conv_block, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # batch_size x in_channels x H x W
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(leakiness, inplace=True)
            # batch_size x out_channels x H' x W'
        )

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
        else:
            output = self.main(inputs)
        return output


class inject_noise(nn.Module):
    def __init__(self, opt, dropout=False):
        super(inject_noise, self).__init__()
        self.ngpu = opt.ngpu
        self.noise_mean = opt.D_noise_mean
        self.noise_stddev = opt.D_noise_stddev
        self.dropout = nn.Sequential()
        if dropout:
            self.dropout = nn.Dropout(opt.D_keep_prob, inplace=True)
        self.noise = torch.FloatTensor()
        if opt.cuda:
            self.noise = self.noise.cuda()

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.dropout, inputs, range(self.ngpu))
            if self.noise_stddev != 0:
                n = Variable(self.noise.resize_(output.size()).normal_(self.noise_mean, self.noise_stddev))
                output += n
        else:
            output = self.dropout(inputs)
            # print(output.size())
            if self.noise_stddev != 0:
                n = Variable(self.noise.resize_(output.size()).normal_(self.noise_mean, self.noise_stddev))
                output += n
        return output


class pixelda_D(nn.Module):
    def __init__(self, in_channels, image_size, opt, kernel_size=3, stride=2, padding=1):
        super(pixelda_D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ngpu = opt.ngpu
        layers, out_channels, projection_size  = self.make_layers(image_size, opt.ndf, opt.D_projection_size, opt)
        self.main = nn.Sequential(
            # batch_size x in_channels x 64 x 64
            nn.Conv2d(in_channels, opt.ndf, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(opt.ndf),
            nn.LeakyReLU(opt.leakiness, inplace=True),
            *layers
        )
        self.fully_connected = nn.Linear(projection_size*projection_size*out_channels, 1)

    def make_layers(self, image_size, filters, projection_size, opt):
        feature_map = image_size[0]     # H or W
        layers = []
        in_channels = filters
        out_channels = in_channels
        while feature_map > projection_size:
            out_channels = in_channels * 2
            for _ in range(1, opt.D_conv_block_size):
                layers.append(conv_block(in_channels, out_channels, ngpu=opt.ngpu,
                              kernel_size=3, stride=1, padding=1, leakiness=opt.leakiness))
                in_channels = out_channels
            layers.append(conv_block(in_channels, out_channels, opt.ngpu,
                          self.kernel_size, self.stride, self.padding, opt.leakiness))
            layers.append(inject_noise(opt, dropout=True))
            in_channels = out_channels
            feature_map = int(np.floor(np.divide(feature_map + 2*self.padding - self.kernel_size, self.stride) + 1))

        assert feature_map == projection_size
        return layers, out_channels, feature_map

    def forward(self, inputs):
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inputs, range(self.ngpu))
            output = output.view(output.size(0), -1)
            output = nn.parallel.data_parallel(self.fully_connected, output, range(self.ngpu))
        else:
            output = self.main(inputs)
            output = output.view(output.size(0), -1)
            output = self.fully_connected(output)
        return output
