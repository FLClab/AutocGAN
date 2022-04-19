# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

def include_each_class(dataset, max_imgs):
    num_each_class = np.zeros(shape=(10,))
    indices = []
    max_per_class = max_imgs / 10
    for i in range(len(dataset)):
        total_imgs = np.sum(num_each_class)
        if total_imgs == max_imgs:
            break
        (img, target) = dataset[i]
        if num_each_class[target] < max_per_class:
            print('Target added: {}'.format(target))
            num_each_class[target] += 1
            indices.append(i)
    return indices, num_each_class


class ImageDataset(object):
    def __init__(self, args, cur_img_size=None):
        img_size = cur_img_size if cur_img_size else args.img_size
        if args.dataset.lower() == "cifar10" or args.dataset.lower() == "cifar100":
            Dt = datasets.CIFAR10
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            args.n_classes = 10
        elif args.dataset.lower() == "stl10":
            Dt = datasets.STL10
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif args.dataset.lower() == "svhn":
            Dt = datasets.SVHN
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif args.dataset.lower() == "imagenet":
            Dt = datasets.ImageNet
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
        elif args.dataset.lower() == "mnist":
            Dt = datasets.MNIST
            args.data_path = './data'
            transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            raise NotImplementedError("Unknown dataset: {}".format(args.dataset))

        if args.dataset.lower() == "stl10" or args.dataset.lower() == "svhn":
            self.train = torch.utils.data.DataLoader(
                Dt(
                    root=args.data_path,
                    split="train",
                    transform=transform,
                    download=True,
                ),
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, split="test", transform=transform, download=True),
                batch_size=args.dis_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.test = self.valid
        else:
            #sub_data = Dt(root=args.data_path, train=True, transform=transform, download=True)
            #size_data = int(50000 * 0.10)
            #indices, num_each_class = include_each_class(sub_data, size_data)
            #print("\nNumber in each class: {}\n".format(num_each_class))
            #sub_data = torch.utils.data.Subset(sub_data, indices)
            self.train = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=True, transform=transform, download=True), # sub_data
                batch_size=args.dis_batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.valid = torch.utils.data.DataLoader(
                Dt(root=args.data_path, train=False, transform=transform),
                batch_size=args.dis_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )

            self.test = self.valid


