from argparse import ArgumentParser
import os
import time
import math
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import pickle

import nets as models
import functions as fns
import data_loader as dataLoader
import common

# Supported data_loaders
data_loader_all = sorted(name for name in dataLoader.__dict__
    if name.islower() and not name.startswith("__")
    and callable(dataLoader.__dict__[name]))

def get_cls_num(dataset):
    # if dataset == 'cifar10': return 10
    # else: 'No idea how many classes!'
    return int(common.DATASET_CLASSES_PARAMS[dataset])

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def compute_topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
def compute_accuracy(output, target):
    output = output.argmax(dim=1)
    acc = 0.0
    acc = torch.sum(target == output).item()
    acc = acc/output.size(0)*100
    return acc


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
            

def main(args):
    master_path = os.path.join(args.dir, 'master')
    worker_path = os.path.join(args.dir, 'worker')
    save_path = os.path.join(master_path, common.MASTER_DATALOADER_FILENAME_TEMPLATE.format(args.dataset))
    data_loader = dataLoader.__dict__[args.dataset](args.dataset_path)
    data_loader.load(save_path)

    if args.model_name == 'ALL':
        all_models = [m for m in os.listdir(master_path) if '.pth.tar' in m]
    else:
        all_models = [args.model_name]
    for model in all_models:
        print('===================================================================')
        print ('Testing model ' + model)
        model = torch.load(os.path.join(master_path, model))
        best_acc = eval(data_loader.get_test_data_loader(), model, args)
        print('Testing accuracy:', best_acc)


if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('dir', type=str, help='path to save models (default: models/)')
    arg_parser.add_argument('model_name', type=str, help='model name to be fl-trained (default: ALL)')
    arg_parser.add_argument('dataset_path', metavar='DIR', help='path to dataset')
    arg_parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    arg_parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='batch size (default: 128)')
    arg_parser.add_argument('--no-cuda', action='store_true', default=False, dest='no_cuda',
                    help='disables training on GPU')
    arg_parser.add_argument('-d', '--dataset',  default='cifar10', 
                        choices=data_loader_all,
                        help='dataset: ' +
                        ' | '.join(data_loader_all) +
                        ' (default: cifar10). Defines which dataset is used. If you want to use your own dataset, please specify here.')
    
    args = arg_parser.parse_args()
    main(args)