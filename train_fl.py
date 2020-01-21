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
import functions as fns
import datetime
import common

# Supported data_loaders
data_loader_all = sorted(name for name in dataLoader.__dict__
    if name.islower() and not name.startswith("__")
    and callable(dataLoader.__dict__[name]))

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_cls_num(dataset):
    # if dataset == 'cifar10': return 10
    # else: 'No idea how many classes!'
    return int(ommon.DATASET_CLASSES_PARAMS[dataset])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def get_avg(self):
        return self.avg
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc

def eval(test_loader, model, args):
    batch_time = AverageMeter()
    acc = AverageMeter()

    # switch to eval mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(test_loader):
        if not args.no_cuda:
            images = images.cuda()
            target = target.cuda()
        output = model(images)
        batch_acc = compute_accuracy(output, target)
        acc.update(batch_acc, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        # Update statistics
        estimated_time_remained = batch_time.get_avg()*(len(test_loader)-i-1)
        fns.update_progress(i, len(test_loader), 
            ESA='{:8.2f}'.format(estimated_time_remained)+'s',
            acc='{:4.2f}'.format(float(batch_acc))
            )
    print()
    print('Test accuracy: {:4.2f}% (time = {:8.2f}s)'.format(
            float(acc.get_avg()), batch_time.get_avg()*len(test_loader)))
    return float(acc.get_avg())


def device_train(train_loader, model, args):
    # switch to train mode
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
        
    for k in range(args.local_epochs):
        for i, (images, target) in enumerate(train_loader):
            target.unsqueeze_(1)
            target_onehot = torch.FloatTensor(target.shape[0], get_cls_num(args.dataset))
            target_onehot.zero_()
            target_onehot.scatter_(1, target, 1)
            target.squeeze_(1)
            
            images = images.cuda()
            target_onehot = target_onehot.cuda()
            target = target.cuda()

            output = model(images)
            loss = criterion(output, target_onehot)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

def run_fl(model_path, data_loader, args, skip_ratio=0.0):
    '''
        Run federated learning
        Note: it competes GPU with worker. TODO: shall not block the gpus
        Input:
            'model_path': fused model's path, waiting to be fl-ed
            'data_loader': data loader for devices
            'fl_epoch': epoch number to run fl on each device
        Output:
    '''
    print ('Run federated learning')

    device_data_idxs = data_loader.device_data_idxs
    model = torch.load(model_path)
    best_acc = 0
    test_loader = data_loader.get_test_data_loader()

    for e in range(args.global_epochs):
        print ('=================================')
        print ('Start global iteration ' + str(e))
        epoch_begin = datetime.datetime.now()

        state_sum = {}
        train_data_num = 0        
        clients = list(np.random.choice([i for i in range(len(device_data_idxs))], args.client_number, replace=False))
        for device_id in clients:
            if device_id % 100 == 0:
                print('Start Device {}'.format(device_id))
            if random.random() < skip_ratio: # skip this device
                continue
            train_loader = data_loader.training_data_loader(device_id)
            device_model = copy.deepcopy(model)
            fine_tuned_model = device_train(train_loader, device_model, args)
            device_state = fine_tuned_model.state_dict()
            device_data_size = data_loader.get_device_data_size(device_id)
            for k in device_state:
                if k not in state_sum:
                    # state_sum[k] = torch.mul(copy.deepcopy(device_state[k]), len(device_data_idxs[device_id]))
                    state_sum[k] = torch.mul(copy.deepcopy(device_state[k]), device_data_size)
                else:
                    state_sum[k] += torch.mul(copy.deepcopy(device_state[k]), device_data_size)
            train_data_num += device_data_size
            del fine_tuned_model
        new_state = {k: torch.div(state_sum[k], train_data_num) for k in state_sum}
        model.load_state_dict(new_state)
        
        acc = eval(test_loader, model, args)
        if acc > best_acc:
            best_acc = acc
            torch.save(model, 'fl_' + model_path)
            print('Save model at epoch: ' + e)

        epoch_end = datetime.datetime.now()
        print ('Epoch ends. Running time: {} mins'.format((epoch_end - epoch_begin).seconds / 60))
        print('Best accuracy:', best_acc)


def main(args):
    master_path = os.path.join(args.dir, 'master')
    worker_path = os.path.join(args.dir, 'worker')
    save_path = os.path.join(master_path, common.MASTER_DATALOADER_FILENAME_TEMPLATE.format(args.dataset))
    data_loader = dataLoader.__dict__[args.dataset](args.dataset_path)
    data_loader.load(save_path)

    if args.model_name == 'ALL':
        all_models = [m for m in os.listdir(master_path) if m.endswith('.pth.tar')]
    else:
        all_models = [args.model_name]
    for model in all_models:
        print ('Federated learning on model ' + model)
        model_path = os.path.join(master_path, model)
        run_fl(model_path, data_loader, args)


if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('dir', type=str, help='path to save models (default: models/)')
    arg_parser.add_argument('model_name', type=str, help='model name to be fl-trained (default: ALL)')
    arg_parser.add_argument('dataset_path', metavar='DIR', help='path to dataset')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    arg_parser.add_argument('-ge', '--global_epochs', default=100, type=int, metavar='N',
                    help='number of total global epochs to run (default: 100)')
    arg_parser.add_argument('-le', '--local_epochs', default=10, type=int, metavar='N',
                    help='number of total local epochs to run (default: 10)')                
    # arg_parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
    #                 help='manual epoch number (useful on restarts)')
    # arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
    #                 choices=model_names,
    #                 help='model architecture: ' +
    #                     ' | '.join(model_names) +
    #                     ' (default: alexnet)')
    arg_parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='batch size (default: 128)')
    arg_parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate (defult: 0.1)', dest='lr')
    arg_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum (default: 0.9)')
    arg_parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)',
                    dest='weight_decay')
    # arg_parser.add_argument('--resume', default='', type=str, metavar='PATH',
    #                 help='path to latest checkpoint (default: none)')
    # arg_parser.add_argument('--no-cuda', action='store_true', default=False, dest='no_cuda',
    #                 help='disables training on GPU')

    # arg_parser.add_argument('-dn', '--device_number', type=int, default=3, 
    #                         help='Total device number.')
    arg_parser.add_argument('-d', '--dataset',  default='cifar10', 
                        choices=data_loader_all,
                        help='dataset: ' +
                        ' | '.join(data_loader_all) +
                        ' (default: cifar10). Defines which dataset is used. If you want to use your own dataset, please specify here.')
    arg_parser.add_argument('-cl', '--client_number', type=int, default=100, 
                            help='Clinet number.')

    args = arg_parser.parse_args()
    main(args)