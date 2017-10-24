import PIL
import os, sys
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from scipy.io import loadmat

class USPS(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform

        file_path = os.path.abspath(os.path.join(data_dir, '../usps_resampled.mat'))
        data = loadmat(file_path)
        X = np.concatenate((data['train_patterns'].T, data['test_patterns'].T))
        Y = np.concatenate((data['train_labels'].T, data['test_labels'].T))

        self.length = X.shape[0]
        dim = int(np.sqrt(X.shape[1]))
        self.mapping = [(X[i].reshape(dim, dim), np.where(Y[i]==1)[0][0]) for i in range(self.length)]

    def __getitem__(self, index):
        (image, label) = self.mapping[index % self.length]
        image = PIL.Image.fromarray(image, 'L')
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return 10*self.length
