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
import json
from .utils.imagenet_helper import *
import io

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


class dataLoader_imagenet_dali(DataLoaderAbstract):

    batch_size = None
    num_workers = None

    train_dataset = None
    val_dataset = None
    device_number = None
    device_data_idxs = None
    device_val_data_idxs = None
    device_belong_to_group = None

    group_number = None
    group_idxs = None
    

    def __init__(self, dataset_path):

        super().__init__()

        # Data loaders.
        self.batch_size = 128
        self.num_workers = 8

        # Data loading code
        # transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        #                          std = [ 0.229, 0.224, 0.225 ]),
        # ])    
        # val_transform = transforms.Compose([
        #     transforms.Scale(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
        #                          std = [ 0.229, 0.224, 0.225 ]),
        # ])
        def _tr(im_resize):
            im_resize = im_resize.convert('RGB')
            buf = io.BytesIO()
            im_resize.save(buf, format='JPEG')
            ret = np.frombuffer(buf.getvalue(), dtype = np.uint8)
            return ret

        transform = transforms.Compose([
            transforms.Lambda(lambda x: _tr(x)),
            ])
        self.dataset_path = dataset_path

        traindir = os.path.join(dataset_path, 'train')
        valdir = os.path.join(dataset_path, 'val')
        self.train_dataset = datasets.ImageFolder(traindir, transform)
        self.val_dataset = datasets.ImageFolder(traindir, transform)
        self.test_dataset = datasets.ImageFolder(valdir, transform)

        # train_loader = torch.utils.data.DataLoader(
        #     train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)        


    def generate_device_data(self, device_number, is_iid):

        '''
            For CIFAR-10 dataset and IID split.
            Since it doesn't have multiple users setting, we split the dataset to device_number parts.
            
        '''

        self.device_number = device_number
        if is_iid:
            num_items = int(len(self.train_dataset)/(self.device_number))
            num_val_items = int(len(self.val_dataset)/(self.device_number))

            self.device_data_idxs, all_idxs = [None for i in range(device_number)], [i for i in range(len(self.train_dataset))]
            self.device_val_data_idxs, all_val_idxs = [None for i in range(device_number)], [i for i in range(len(self.val_dataset))]

            for i in range(self.device_number):
                self.device_data_idxs[i] = list(np.random.choice(all_idxs, num_items, replace=False))
                all_idxs = list(set(all_idxs) - set(self.device_data_idxs[i]))
                self.device_val_data_idxs[i] = list(np.random.choice(all_val_idxs, num_val_items, replace=False))
                all_val_idxs = list(set(all_val_idxs) - set(self.device_val_data_idxs[i]))
        else:
            #TODO(zhaoyx): non-IID for imagenet.
            raise ValueError("need be iid for imagenet.")

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
            shuffle=True, num_workers=self.num_workers, pin_memory=True, collate_fn=collate_fn)

        return get_imagenet_iter_dali_with_custom_dataloader(train_loader, self.batch_size, self.num_workers)


    def validation_data_loader(self, device_idx):
        '''
            Generate validation data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `validation_data_loader`: validation data loader for the device 
        '''

        val_loader = torch.utils.data.DataLoader(
            DatasetSplit(self.val_dataset, self.device_val_data_idxs[device_idx]), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn)

        return get_imagenet_iter_dali_with_custom_dataloader(val_loader, self.batch_size, self.num_workers)

    def validation_data_loader_of_devices(self, device_idxs):
        '''
            Generate validation data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `validation_data_loader`: validation data loader for the device 
        '''
        array = []
        for device_idx in device_idxs:
            array.extend(self.device_val_data_idxs[device_idx])

        val_loader = torch.utils.data.DataLoader(
            DatasetSplit(self.val_dataset, array), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True, collate_fn=collate_fn)

        return get_imagenet_iter_dali_with_custom_dataloader(val_loader, self.batch_size, self.num_workers)


    def dump(self, save_path):
        '''
            Generate validation data loader for specified device.
            Input:
                `group_idxs`: (List)
                `device_data_idxs`: (List)
        '''
        for i in range(len(self.device_data_idxs)):
            self.device_data_idxs[i] = [int(j) for j in self.device_data_idxs[i]]
            self.device_val_data_idxs[i] = [int(j) for j in self.device_val_data_idxs[i]]
        save_dict = {'device_data_idxs' : self.device_data_idxs, 
                     'device_val_data_idxs' : self.device_val_data_idxs,
                     'group_idxs' : self.group_idxs}
        
        with open(save_path, 'w') as file_id:
            json.dump(save_dict, file_id)


    def load(self, save_path):
        '''
            Generate validation data loader for specified device.
            Input:
                `group_idxs`: (List)
                `device_data_idxs`: (List)
        '''
        with open(save_path, 'r') as file_id:
            read_dict = json.load(file_id)

        self.group_idxs = read_dict['group_idxs']
        self.device_data_idxs = read_dict['device_data_idxs']
        self.device_val_data_idxs = read_dict['device_val_data_idxs']
        self.device_number = len(self.device_data_idxs)
        self.group_number = len(self.group_idxs)

    def get_all_train_data_loader(self):

        train_loader = get_imagenet_iter_dali(type='train', image_dir=self.dataset_path, batch_size=self.batch_size,
                                          num_threads=self.num_workers, crop=224, device_id=0, num_gpus=1)

        return train_loader

    def get_all_validation_data_loader(self):
        '''
            Generate validation data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `validation_data_loader`: validation data loader for the device 
        '''

        val_loader = get_imagenet_iter_dali(type='train-val', image_dir=self.dataset_path, batch_size=self.batch_size,
                                          num_threads=self.num_workers, crop=224, device_id=0, num_gpus=1)

        return val_loader

    def get_test_data_loader(self):


        test_loader = get_imagenet_iter_dali(type='val', image_dir=self.dataset_path, batch_size=self.batch_size,
                                          num_threads=self.num_workers, crop=224, device_id=0, num_gpus=1)


        return test_loader

    def get_device_data_size(self,device_idx):
        return len(self.device_data_idxs[device_idx])

    def get_device_val_data_size(self, device_idx):

        return int(len(self.val_dataset)/(self.device_number))



def imagenet_dali(dataset_path):
    return dataLoader_imagenet_dali(dataset_path)

