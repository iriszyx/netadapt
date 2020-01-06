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

sys.path.append(os.path.abspath('../'))

from constants import *

class DatasetUserSplit(Dataset):
    def __init__(self, dataset, user):
        self.dataset = dataset[user]

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, index):
        image, label = self.dataset['x'][index], self.dataset['y'][index]
        image = np.resize(np.array(image),(28,28))
        image = Image.fromarray(image)

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        image = transform(image)

        return image, label



class trainset(Dataset):
    def __init__(self, train_data, users):
        
        self.dataset = {'x':[], 'y':[]}
        
        self.users = users
        
        for user in self.users:
        	self.dataset['x'].extend(train_data[user]['x'])
        	self.dataset['y'].extend(train_data[user]['y'])

    def __getitem__(self, index):
        image, label = self.dataset['x'][index], self.dataset['y'][index]
        image = np.resize(np.array(image),(28,28))
        image = Image.fromarray(image)
        

        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        image = transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset['x'])


class dataLoader_feminst(DataLoaderAbstract):

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

        train_data_dir = os.path.join(dataset_path, 'feminst','train')
        test_data_dir = os.path.join(dataset_path,'feminst','test')

        users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

        self.dataset_users = users       
        self.train_dataset = train_data
        self.val_dataset = test_data

 


    def generate_device_data(self, device_number, is_iid):

        '''
            For Feminst dataset and non-IID split.
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
            #TODO(zhaoyx): IID for Feminst.
            raise ValueError("need be non-iid for Feminst.")

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
           For 
        '''
        #TODO(zhaoyx): sort and greedy
        self.group_number = group_number
        self.group_idxs = [[] for i in range(self.group_number)]

        self.device_data_size = [len(self.train_dataset[i]['x']) for i in self.device_data_idxs]

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
            DatasetUserSplit(self.train_dataset, user_name), batch_size=self.batch_size, 
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
            DatasetUserSplit(self.val_dataset, user_name), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=False)

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
            trainset(self.train_dataset, self.dataset_users), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=True)

        return train_loader

    def get_all_validation_data_loader(self):

        val_loader = torch.utils.data.DataLoader(
            trainset(self.val_dataset, self.dataset_users), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=False)

        return val_loader

    def get_test_data_loader(self):

        val_loader = torch.utils.data.DataLoader(
            trainset(self.val_dataset, self.dataset_users), batch_size=self.batch_size, 
            num_workers=self.num_workers, pin_memory=True, shuffle=False)

        return val_loader

    def get_device_data_size(self, device_idx):
        user_name = self.dataset_users[self.device_data_idxs[device_idx]]

        return len(self.train_dataset[user_name]['x'])



def feminst(dataset_path):
    return dataLoader_feminst(dataset_path)


