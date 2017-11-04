import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import matplotlib.pyplot as plt


class MNIST_M(torch.utils.data.Dataset):
    def __init__(self, root_dir, train, transform=None):
        self.root_dir = root_dir
        self.train = train
        self.transform = transform
        if (train):
            self.image_dir = os.path.join(self.root_dir, 'mnist_m_train')
            labels_file = os.path.join(self.root_dir, "mnist_m_train_labels.txt")
        else:
            self.image_dir = os.path.join(self.root_dir, 'mnist_m_test')
            labels_file = os.path.join(self.root_dir, "mnist_m_test_labels.txt")

        with open(labels_file, "r") as fp:
        	content = fp.readlines()
        self.mapping = list(map(lambda x: (x[0], int(x[1])), [c.strip().split() for c in content]))

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        image, label = self.mapping[idx]
        image = os.path.join(self.image_dir, image)
        print(np.array(Image.open(image).convert('RGB').getdata()).shape)
        image = self.transform(Image.open(image).convert('RGB'))
        return image, label


if __name__ == '__main__':
	root_dir = 'data/mnist_m'
	batch_size = 10
	composed_transform = transforms.Compose([transforms.Scale((224,224)),transforms.ToTensor()])
	train_dataset = MNIST_M(root_dir=root_dir, train=True, transform=composed_transform)
	test_dataset = MNIST_M(root_dir=root_dir, train=False, transform=composed_transform)

	print('Size of train dataset: %d' % len(train_dataset))
	print('Size of test dataset: %d' % len(test_dataset))

	train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

	def imshow(img):
	    npimg = img.numpy()
	    plt.imshow(np.transpose(npimg, (1, 2, 0)))
	    # plt.show()

	train_dataiter = iter(train_loader)
	train_images, train_labels = train_dataiter.next()
	print("Train images", train_images)
	print("Train images", train_labels)
	imshow(torchvision.utils.make_grid(train_images))

	test_dataiter = iter(test_loader)
	test_images, test_labels = test_dataiter.next()
	print("Test images", test_labels)
	imshow(torchvision.utils.make_grid(test_images))
