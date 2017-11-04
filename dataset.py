import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms


## Import Data Loaders ##
from mnistm_loader import *
from usps_loader import *


def get_dataset(dataset, root_dir, imageSize, batchSize, workers=1):
    if dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=root_dir, download=True, train=True,
                                      transform=transforms.Compose([
                                      transforms.Scale(imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
        test_dataset = dset.CIFAR10(root=root_dir, download=True, train=False,
                                      transform=transforms.Compose([
                                      transforms.Scale(imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
    elif dataset == 'mnist':
        train_dataset = dset.MNIST(root=root_dir, train=True, download=True,
                                    transform=transforms.Compose([
                                    transforms.Scale(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        test_dataset = dset.MNIST(root=root_dir, train=False, download=True,
                                    transform=transforms.Compose([
                                    transforms.Scale(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
    elif dataset == 'mnistm':
        train_dataset = MNIST_M(data_dir=root_dir, train=True,
                                 transform=transforms.Compose([
                                 transforms.Scale(imageSize),
                                 transforms.ToTensor(),
                                 ]))
        test_dataset = MNIST_M(data_dir=root_dir, train=False,
                                 transform=transforms.Compose([
                                 transforms.Scale(imageSize),
                                 transforms.ToTensor(),
                                 ]))
    elif dataset == 'usps':
        train_dataset = USPS(data_dir=root_dir, train=True,
                              image_size=imageSize,
                              transform=transforms.Compose([
                              transforms.Scale(imageSize),
                              ]))
        test_dataset = USPS(data_dir=root_dir, train=False,
                              image_size=imageSize,
                              transform=transforms.Compose([
                              transforms.Scale(imageSize),
                              ]))

    assert train_dataset, test_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize,
                                                   shuffle=True, num_workers=int(workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader
