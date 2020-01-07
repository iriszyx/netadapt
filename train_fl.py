from argparse import ArgumentParser
import os
import pickle
import time
import torch
from shutil import copyfile
import subprocess
import sys
import warnings
import common
import network_utils as networkUtils
import data_loader as dataLoader
import copy
import numpy as np
import functools
import random

def run_fl(model_path, data_loader, network_utils, fl_epoch, skip_ratio=0.0):
    '''
        Run federated learning
        Note: it competes GPU with worker. TODO: shall not block the gpus
        Input:
            'model_path': fused model's path, waiting to be fl-ed
            'data_loader': data loader for devices
            'network_utils': network util
            'fl_epoch': epoch number to run fl on each device
        Output:
            'fl_model_path': a new model path where fl-ed model is stored
    '''
    print ('Run federated learning')
    os.environ["CUDA_VISIBLE_DEVICES"] = 0
    state_sum = {}
    train_data_num = 0
    device_data_idxs = data_loader.device_data_idxs
    new_model_path = model_path + '_fl'
    model = torch.load(model_path)
    for device_id in range(len(device_data_idxs)):
        if random.random() < skip_ratio: # skip this device
            continue
        train_loader = data_loader.training_data_loader(device_id)
        device_model = copy.deepcopy(model)
        fine_tuned_model = network_utils.fine_tune(device_model, fl_epoch, train_loader)
        device_state = fine_tuned_model.state_dict()
        for k in device_state:
            if k not in state_sum:
                state_sum[k] = torch.mul(copy.deepcopy(device_state[k]), len(device_data_idxs[device_id]))
            else:
                state_sum[k] += torch.mul(copy.deepcopy(device_state[k]), len(device_data_idxs[device_id]))
        train_data_num += len(device_data_idxs[device_id])
        del fine_tuned_model
    new_state = torch.div(state_sum, train_data_num)
    model.load_state_dict(new_state)
    torch.save(model, new_model_path)
    return model_path