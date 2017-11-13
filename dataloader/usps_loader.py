import PIL
import os, sys
import numpy as np
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pickle

class USPS(torch.utils.data.Dataset):
    def __init__(self, root, train, image_size, transform=None, split="original"):
        self.transform = transform
        self.train = train
        self.image_size = (image_size, image_size) if type(image_size) is int else image_size
        file_path = os.path.abspath(os.path.join(root, 'usps_resampled.mat'))
        data = loadmat(file_path)
        if split == "original":
            # 7291 train and 2007 test split
            split_file = os.path.abspath(os.path.join(root, 'usps_split.pkl'))
            X = np.concatenate((data['train_patterns'].T, data['test_patterns'].T))
            Y = np.concatenate((data['train_labels'].T, data['test_labels'].T))
            with open(split_file, "rb") as f:
                split_dict = pickle.load(f)
            if train:
                X = X[split_dict['train_split']]
                Y = Y[split_dict['train_split']]
            else:
                X = X[split_dict['test_split']]
                Y = Y[split_dict['test_split']]
        else:
            # 50%-50% split
            if train:
                X = data['train_patterns'].T
                Y = data['train_labels'].T
            else:
                X = data['test_patterns'].T
                Y = data['test_labels'].T
        self.length = X.shape[0]
        dim = int(np.sqrt(X.shape[1]))
        self.mapping = [(X[i].reshape(dim, dim), np.where(Y[i]==1)[0][0]) for i in range(self.length)]

    def __getitem__(self, index):
        (image, label) = self.mapping[index]
        image = PIL.Image.fromarray(image)
        if self.transform is not None:
            image = self.transform(image)
        image = torch.FloatTensor(np.array(image.getdata()))
        image.resize_(1, *self.image_size)
        return image, label

    def __len__(self):
        return self.length

if __name__ == '__main__':
	root_dir = '../data/usps'
	batch_size = 10
	composed_transform = transforms.Compose([transforms.Scale(64)])
	train_dataset = USPS(root=root_dir, train=True, image_size=(64,64), transform=composed_transform)
	test_dataset = USPS(root=root_dir, train=False, image_size=(64,64), transform=composed_transform)

	print('Size of train dataset: %d' % len(train_dataset))
	print('Size of test dataset: %d' % len(test_dataset))

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	print('Size of train dataset: %d' % len(train_loader))
	print('Size of test dataset: %d' % len(test_loader))

	def imshow(img):
	    npimg = img.numpy()
	    plt.imshow(np.transpose(npimg, (1, 2, 0)))
	    plt.show()

	train_dataiter = iter(train_loader)
	train_images, train_labels = train_dataiter.next()
	print("Train images", train_images)
	print("Train images", train_labels)
	imshow(torchvision.utils.make_grid(train_images))


## TRASH ##
# Code to split USPS data
# from sklearn.model_selection import StratifiedShuffleSplit as SSS
# sss= SSS(n_splits=1, test_size=2007./self.length, random_state=9926)
# y = np.array([np.where(Y[i]==1)[0][0] for i in range(self.length)])
# print(list(sss.split(X, Y)))
# train_idx, test_idx = list(sss.split(X, y))[0]
# print(train_idx.shape)
# print(test_idx.shape)
# d = dict(train_split=train_idx, test_split=test_idx)
# with open("usps_split.pkl", "wb") as f:
#     pickle.dump(d, f)
