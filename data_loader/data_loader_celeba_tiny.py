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
from .utils.model_utils import *
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from PIL import Image
import heapq
# test 

sys.path.append(os.path.abspath('../'))

from constants import *

class TrainDatasetUserSplit(Dataset):
    def __init__(self, dataset, user, dataset_path):
        self.dataset = dataset[user]
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dataset_path, 'celeba', 'img_align_celeba', self.dataset['x'][index]))
        image = image.resize((128, 128)).convert('RGB')

        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4), 
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)

        #label = torch.Tensor(int(self.dataset['y'][index]))
        label = int(self.dataset['y'][index])

        return image, label

class TestDatasetUserSplit(Dataset):
    def __init__(self, dataset, user, dataset_path):
        self.dataset = dataset[user]
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dataset_path, 'celeba', 'img_align_celeba', self.dataset['x'][index]))
        # image = image.resize((128, 128)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)

        label = int(self.dataset['y'][index])

        return image, label


class trainset(Dataset):
    def __init__(self, train_data, users, dataset_path):
        
        self.dataset = {'x':[], 'y':[]}
        
        self.users = users
        
        for user in self.users:
        	self.dataset['x'].extend(train_data[user]['x'])
        	self.dataset['y'].extend(train_data[user]['y'])

        self.dataset_path = dataset_path

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dataset_path, 'celeba', 'img_align_celeba', self.dataset['x'][index]))
        # image = image.resize((128, 128)).convert('RGB')

        transform = transforms.Compose([
            # transforms.RandomCrop(32, padding=4), 
            transforms.Resize(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)

        label = int(self.dataset['y'][index])

        return image, label


    def __len__(self):
        return len(self.dataset['x'])

class testset(Dataset):
    def __init__(self, train_data, users, dataset_path):
        
        self.dataset = {'x':[], 'y':[]}
        
        self.users = users
        
        for user in self.users:
            self.dataset['x'].extend(train_data[user]['x'])
            self.dataset['y'].extend(train_data[user]['y'])

        self.dataset_path = dataset_path

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.dataset_path, 'celeba', 'img_align_celeba', self.dataset['x'][index]))
        # image = image.resize((128, 128)).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        image = transform(image)

        label = int(self.dataset['y'][index])

        return image, label

    def __len__(self):
        return len(self.dataset['x'])


class dataLoader_celeba_tiny(DataLoaderAbstract):

    batch_size = None
    num_workers = None

    train_dataset = None
    val_dataset = None
    device_number = None
    device_data_idxs = None
    device_belong_to_group = None

    group_number = None
    group_idxs = None
    dataset_path = None

    def __init__(self, dataset_path):

        super().__init__()

        # Data loaders.
        self.batch_size = 128
        self.num_workers = 8

        train_data_dir = os.path.join(dataset_path, 'celeba','train')
        test_data_dir = os.path.join(dataset_path,'celeba','test')

        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

        self.dataset_users = users       
        self.train_dataset = train_data
        self.val_dataset = train_data
        self.test_dataset = test_data
        self.dataset_path = dataset_path



 


    def generate_device_data(self, device_number, is_iid):

        '''
            For Celeba dataset and non-IID split.
            If users number < device_number, resample some users.
            If users number >= device_number, sample device_number from users.
        '''
        user_number = len(self.dataset_users)
        self.device_number = device_number
        if not is_iid:
            self.device_data_idxs = [None for i in range(device_number)]

            if user_number < self.device_number:
                #resample
                self.device_data_idxs = list(np.random.choice([i for i in range(user_number)], self.device_number, replace=True))
                
            else:
                #sample
                self.device_data_idxs = list(np.random.choice([i for i in range(user_number)], self.device_number, replace=False))
            
        else:
            #TODO(zhaoyx): IID for Celeba.
            raise ValueError("need be non-iid for Celeba.")

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
        #self.device_belong_to_group = [None for i in range(len(self.train_dataset))]

        for i in range(self.device_number):
            self.group_idxs[i%self.group_number].append(i)
            #self.device_belong_to_group[i] = i%self.group_number

        return self.group_idxs


    def generate_group_based_on_device_data_size(self, group_number):        
        '''
           Generate group of devices with approximate total device data size.
        '''
        #TODO(zhaoyx): sort and greedy
        self.group_number = group_number
        self.group_idxs = [[] for i in range(self.group_number)]

        self.device_data_size = [[len(self.train_dataset[self.dataset_users[self.device_data_idxs[i]]]['x']), i] 
                                    for i in range(len(self.device_data_idxs))]
        self.device_data_size = sorted(self.device_data_size, key=lambda x: x[0], reverse = True)
        self.heap = [[0,i] for i in range(group_number)]

        for i in self.device_data_size:
            now_data_size = i[0]
            now_data_index = i[1]

            group_data_size, group_idx = heapq.heappop(self.heap)
            group_data_size = group_data_size + now_data_size
            self.group_idxs[group_idx].append(now_data_index)
            heapq.heappush(self.heap, [group_data_size, group_idx])

        return self.group_idxs


    def training_data_loader(self, device_idx):
        '''
            Generate traing data loader for specified device.
            Input:
                `device_idx`: (int) index of the device

            Output:
                `training_data_loader`: training data loader for the device 
        '''
        user_name = self.dataset_users[self.device_data_idxs[device_idx]]
        if self.get_device_data_size(device_idx) == 0:
            return None

        train_loader = torch.utils.data.DataLoader(
            TrainDatasetUserSplit(self.train_dataset, user_name, self.dataset_path), batch_size=self.batch_size, 
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

        user_name = self.dataset_users[self.device_data_idxs[device_idx]]

        val_loader = torch.utils.data.DataLoader(
            TestDatasetUserSplit(self.val_dataset, user_name, self.dataset_path), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True)

        return val_loader

    def dump(self, save_path):
        '''
            Generate validation data loader for specified device.
            Input:
                `group_idxs`: (List)
                `device_data_idxs`: (List)
        '''
        for i in range(len(self.device_data_idxs)):
            self.device_data_idxs[i] = int(self.device_data_idxs[i])

        save_dict = {'device_data_idxs' : self.device_data_idxs, 
                     'group_idxs' : self.group_idxs,
                     'dataset_users' : self.dataset_users}
        
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
        self.dataset_users = read_dict['dataset_users']
        self.device_number = len(self.device_data_idxs)
        self.group_number = len(self.group_idxs)

    def get_all_train_data_loader(self):

        train_loader = torch.utils.data.DataLoader(
            trainset(self.train_dataset, self.dataset_users, self.dataset_path), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True)

        return train_loader

    def get_all_validation_data_loader(self):

        val_loader = torch.utils.data.DataLoader(
            testset(self.val_dataset, self.dataset_users, self.dataset_path), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True)

        return val_loader

    def get_test_data_loader(self):

        val_loader = torch.utils.data.DataLoader(
            testset(self.test_dataset, self.dataset_users, self.dataset_path), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=False)

        return val_loader

    def get_device_data_size(self, device_idx):
        user_name = self.dataset_users[self.device_data_idxs[device_idx]]

        return len(self.train_dataset[user_name]['x'])

    def get_device_val_data_size(self, device_idx):

        user_name = self.dataset_users[self.device_data_idxs[device_idx]]

        return len(self.val_dataset[user_name]['x'])



def celeba_tiny(dataset_path):
    return dataLoader_celeba_tiny(dataset_path)
