'''
Convolutional MNIST model
'''

import torch
import torch.nn as nn
import numpy as np


class MnistClassifier(nn.Module):
    def __init__(self, source_channels, target_channels, num_classes, ngpu=1):
        super(MnistClassifier, self).__init__()
        self.ngpu = ngpu
        self.private_source = nn.Sequential(
            # batch_size x source_channels x 64 x 64
            nn.Conv2d(source_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.Relu(True),
            # batch_size x 32 x 64 x 64
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # batch_size x 32 x 32 x 32
        )
        self.private_target = nn.Sequential(
            # batch_size x source_channels x 64 x 64
            nn.Conv2d(target_channels, 32, kernel_size=5, stride=1, padding=2),
            nn.Relu(True),
            # batch_size x 32 x 64 x 64
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            # batch_size x 32 x 32 x 32
        )
        self.shared_convs = nn.Sequential(
            # batch_size x 32 x 32 x 32
            nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),
            nn.Relu(True),
            # batch_size x 48 x 32 x 32
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            # batch_size x 48 x 16 x 16
        )
        self.shared_fcs = nn.Sequential(
            # batch_size x (48*16*16 = 12288)
            nn.Linear(12288, 100),
            nn.Relu(True),
            # batch_size x 100
            nn.Linear(100, 100),
            nn.Relu(True),
            # batch_size x 100
            nn.Linear(100, num_classes)
        )


    def forward(self, inputs, dataset="target"):
        if dataset == "target":
            private_net = self.private_target
        else:
            private_net = self.private_source
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(private_net, inputs, range(self.ngpu))
            output = nn.parallel.data_parallel(self.shared_convs, output, range(self.ngpu))
            output = output.view(output.size(0), -1)
            output = nn.parallel.data_parallel(self.shared_fcs, output, range(self.ngpu))
        else:
            output = private_net(inputs)
            output = self.shared_convs(output)
            output = output.view(output.size(0), -1)
            output = self.shared_fcs(output)
        return output
