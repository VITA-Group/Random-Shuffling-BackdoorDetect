import sys
import cv2
import torch
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from torchvision.datasets import CIFAR10

from torch.utils.data import DataLoader, Subset

class PatchedCIFAR10_mt(data.Dataset):
    def __init__(self, root, mode,
                poison_ratio=0.05, target_num=0, patch_size=5,
                augmentation=True, use_normalize=True):

        print('Multiple Targets: trigger_i -> class_i, target class num: %d'%target_num)

        self.poison_ratio = poison_ratio
        self.root = root

        trigger_list = []
        for i in range(target_num):
            try:
                trigger = torch.tensor(cv2.imread("dataset/triggers/multiple_triggers/mask0{}.bmp".format(i))).unsqueeze(0)
            except:
                trigger = torch.tensor(cv2.imread("../../dataset/triggers/multiple_triggers/mask0{}.bmp".format(i))).unsqueeze(0)
            trigger_list.append(trigger)

        normalize = transforms.Normalize(mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010))

        if augmentation and mode == 'train':
            transform_list = [
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        else:
            transform_list = [
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                ]
        if use_normalize:
            print('Contain Normalization in data-augumentation')
            transform_list.append(normalize)
        else:
            print('Contain no Normalization in data-augumentation')
        
        self.transform = transforms.Compose(transform_list)
        if mode == 'test':
            dataset = CIFAR10(root, train=False, transform=self.transform, download=True)
            self.imgs = dataset.data
            self.labels = torch.tensor(dataset.targets)
        elif mode == 'train' or mode == 'val':
            dataset = CIFAR10(root, train=True, transform=self.transform, download=True)
            if mode == 'train':
                self.imgs = dataset.data[:45000]
                self.labels = torch.tensor(dataset.targets[:45000])
            else:
                self.imgs = dataset.data[45000:]
                self.labels = torch.tensor(dataset.targets[45000:])
        else:
            assert False

        image_size = self.imgs.shape[1]
        
        if mode == 'train':
            subset_choose = np.random.choice(list(range(len(self.imgs))), size=int(poison_ratio*len(self.imgs)*target_num), replace=False)
        else:
            poison_ratio = 1 / target_num
            subset_choose = np.random.choice(list(range(len(self.imgs))), size=len(self.imgs), replace=False)   

        end = 0
        start_x = image_size - patch_size - 3
        start_y = image_size - patch_size - 3

        for target_class in range(target_num):
            start = end
            end = int((target_class+1)*poison_ratio*len(self.imgs))
            self.imgs[subset_choose[start:end], start_x: start_x + patch_size, start_y: start_y + patch_size, :] = trigger_list[target_class]
            self.labels[subset_choose[start:end]] = target_class

            print('Poison %d Images to class %d'%(end-start, target_class)) 
        
        self.imgs = torch.tensor(np.transpose(self.imgs, (0,3,1,2))) # Batch-size, 3, 32, 32

    def __getitem__(self, index):
        img = self.transform(self.imgs[index])
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)
