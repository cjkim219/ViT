import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


class MNIST_dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def data_loader(self):
        print("MNIST data load")

        train_set = torchvision.datasets.MNIST(
        root = './data/MNIST',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip() 
        ])
        )
        test_set = torchvision.datasets.MNIST(
        
            root = './data/MNIST',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor() 
            ])
        )

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        
        return train_loader, test_loader



class CIFAR10_dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def data_loader(self):
        print("CIFAR10 data load")

        train_set = torchvision.datasets.CIFAR10(
        root = './data/CIFAR10',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
        ])
        )
        test_set = torchvision.datasets.CIFAR10(
            root = './data/CIFAR10',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        )

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        
        return train_loader, test_loader



class CIFAR100_dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def data_loader(self):
        print("CIFAR100 data load")

        train_set = torchvision.datasets.CIFAR100(
        root = './data/CIFAR100',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip()
        ])
        )
        test_set = torchvision.datasets.CIFAR100(
            root = './data/CIFAR100',
            train = False,
            download = True,
            transform = transforms.Compose([
                transforms.ToTensor()
            ])
        )

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size)
        
        return train_loader, test_loader
    


class ImageNet_dataset():
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def data_loader(self):
        print("ImageNet data load")
        
        train_set = torchvision.datasets.ImageFolder(
            root = '~/JH/Data/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/train',
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip()
            ])
        )
        val_set = torchvision.datasets.ImageFolder(
            root = '~/JH/Data/imagenet_object_localization_patched2019/ILSVRC/Data/CLS-LOC/val',
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        )

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size)
        
        return train_loader, val_loader