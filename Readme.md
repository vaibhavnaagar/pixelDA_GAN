# Unsupervised Domain Adaptation with GAN #
> CS698U (Visual Recognition)

* PyTorch implementation of paper- [Bousmalis, Konstantinos, et al. "Unsupervised Pixel-level Domain Adaptation with GANs." (2017)](https://arxiv.org/abs/1612.05424)

* All classifiers except `mnist_classifer.py` are taken from [https://github.com/kuangliu/pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)

## Folder Structure
Basic folder structure
```
├── pixelda_gan_classifier.py # PIXELDA GAN (as per the paper)
├── pixelda_lsgan_classifier.py # Same architecture as of PIXELDA GAN with modified loss function similar to LS-GAN
├── dcgan_classifier.py # DCGAN architecture
├── run_classifier.py # Resume training or test a classifer
├── params.py # Model parameters
├── dataset.py # create dataloaders
├── plotter.py # generate plots parallelly
├── utils.py # utils
├── dataloader # custom dataloaders
│   ├── mnistm_loader.py
|   └── usps_loader.py
├── classifiers # Classifiers' architecture
│   ├── mnist_classifier.py # Shared layered classifier (as per the paper)
|   └── *.py # Other classifiers
├── GANs # GAN architecture
│   ├── dcgan.py # DC-GAN architecture
|   └── pixelda_gan.py # PIXELDA-GAN architecture
├── data (not included in the repo)
│   ├── mnist # mnist data (not included in the repo); subdirectories will be created by pytorch (using torchvision.datasets)
│   |   ├── processed
│           ├── test.pt
│           └── training.pt
│   |   └── raw
│           ├── t10k-images-idx3-ubyte
│           ├── t10k-labels-idx1-ubyte
│           ├── train-images-idx3-ubyte
│           └── train-labels-idx1-ubyte
│   ├── mnist_m # MNIST-M dataset (not included in the repo)
        (Download: https://drive.google.com/drive/folders/0B_tExHiYS-0vR2dNZEU4NGlSSW8)
│   |   ├── mnist_m_test
│           └── *.png
│   |   ├── mnist_m_train
│           └── *.png
│       ├── mnist_m_test_labels.txt
│       ├── mnist_m_train_labels.txt
│   └── usps # USPS dataset (not included in the repo)
        (Download: https://github.com/marionmari/Graph_stuff/tree/master/usps_digit_data)
│       ├── usps_resampled.mat
│       └── usps_split.pkl # created by code
├── checkpoint # model files to be saved here (not included in the repo)
├── images # generated images to be saved here (not included in the repo)
└── plots # generated plots to be saved here (not included in the repo)
```

## Requirements
* PyTorch (from [source](https://github.com/pytorch/pytorch#from-source))
* cuda 8.0
* NVIDIA GTX
* Python 3.6.2
* matplotlib
* numpy
* multiprocessing
