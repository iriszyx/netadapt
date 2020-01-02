from .data_loader_abstract import DataLoaderAbstract
import torch
import os
import sys
import copy
import time
import torch
import pickle
import warnings
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.sampler as sampler
from torch.utils.data import DataLoader, Dataset
import numpy as np

sys.path.append(os.path.abspath('../'))

from constants import *

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class dataLoader_cifar10(DataLoaderAbstract):

    batch_size = None
    num_workers = None

    train_dataset = None
    val_dataset = None
    device_number = None
    device_data_idxs = None
    device_belong_to_group = None

    group_number = None
    group_idxs = None
    

    def __init__(self, dataset_path):

        super().__init__()

        # Data loaders.
        self.batch_size = 128
        self.num_workers = 4
        
        self.train_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))
        
        self.val_dataset = datasets.CIFAR10(root=dataset_path, train=True, download=True,
        transform=transforms.Compose([
            transforms.Resize(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]))


    def generate_device_data(self, device_number, is_iid):

        '''
            For CIFAR-10 dataset and IID split.
            Since it doesn't have multiple users setting, we split the dataset to device_number parts.
            
        '''

        self.device_number = device_number
        if is_iid:
            num_items = int(len(self.train_dataset)/(self.device_number))

            self.device_data_idxs, all_idxs = [None for i in range(device_number)], [i for i in range(len(self.train_dataset))]

            for i in range(self.device_number):
                self.device_data_idxs[i] = set(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - self.device_data_idxs[i])
        else:
            #TODO(zhaoyx): non-IID for cifar-10.
            raise ValueError("need be iid for cifar-10.")

        return self.device_data_idxs


    def generate_group_based_on_device_number(self, group_number):
        '''
            Generate group of devices with approximate total device number.
            
            Input:
                `group_number`: (int) group number 

            Output:
                `group_idxs`: 
            
        '''
        self.group_number = group_number
        self.group_idxs = [[] for i in range(self.group_number)]
        #TODO(zhaoyx): dict or list
        self.device_belong_to_group = [None for i in range(len(self.train_dataset))]

        for i in range(self.device_number):
            self.group_idxs[i%self.group_number].append(i)
            self.device_belong_to_group[i] = i%self.group_number

        return self.group_idxs


    def generate_group_based_on_device_data_size(self, group_number):        
        '''
           For CIFAR-10 with IID spliting, same as `generate_group_based_on_device_number`.
        '''
        return self.generate_group_based_on_device_number(group_number)


    def training_data_loader(self, device_idx):
        '''
            Generate traing data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `training_data_loader`: training data loader for the device 
        '''
        train_loader = torch.utils.data.DataLoader(
            DatasetSplit(self.train_dataset, self.device_data_idxs[device_idx]), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True)

        return train_loader


    def validation_data_loader(self, device_idx):
        '''
            Generate validation data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `validation_data_loader`: validation data loader for the device 
        '''

        val_loader = torch.utils.data.DataLoader(
            DatasetSplit(self.val_dataset, self.device_data_idxs[device_idx]), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=False)

        return val_loader


    def load(self, group_idxs, device_data_idxs):
        '''
            Generate validation data loader for specified device.
            Input:
                `group_idxs`: (List)
                `device_data_idxs`: (List)
        '''
        self.group_idxs = group_idxs
        self.device_data_idxs = device_data_idxs
        self.device_number = len(device_data_idxs)
        self.group_number = len(group_idxs)



def cifar10(dataset_path):
    return dataLoader_cifar10(dataset_path)

