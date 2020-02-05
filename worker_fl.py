from argparse import ArgumentParser
import torch
import os
import common 
import constants 
import copy
import network_utils as networkUtils
import data_loader as dataLoader
import numpy as np
import datetime
import functools
import random

'''
    Launched by `master.py`
    
    Simplify a certain block of models and then finetune for several iterations.
'''


# Supported network_utils
network_utils_all = sorted(name for name in networkUtils.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networkUtils.__dict__[name]))

# Supported data_loaders
data_loader_all = sorted(name for name in dataLoader.__dict__
    if name.islower() and not name.startswith("__")
    and callable(dataLoader.__dict__[name]))

def _fed_avg(w, data_num):
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    w_avg = {k: [torch.mul(w[i][k], data_num[i]).cpu() for i in range(len(w))] for k in w[0]}
    w_avg = {k: functools.reduce(lambda x,y: x + y, w_avg[k]) / np.sum(data_num) for k in w_avg}
  
    return w_avg

def _model_fusion(worker_folder, iteration, block_idx, device_num, data_num):
    '''
        Fuse model generate from different devices from best_block worker.
        
        Input:
            `worker_folder`: (string) directory where `worker.py` will save models.
            `iteration`: (int) NetAdapt iteration.
            `block_idx`: (int) block index of the best network.
            `device_num`: device numbers to be fused.
            

        Output:
            `best_model_path`: (string) path to the best model generated from model fusion.
    '''

    w_array = []

    for device_idx in range(device_num):
        model_path = os.path.join(worker_folder,
                                  common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, block_idx, device_idx))
        model = torch.load(model_path)
        s1 = model.state_dict()
        w_array.append(copy.deepcopy(s1))
        del model
        del s1

    w_glob = _fed_avg(w_array, data_num)

    model_path = os.path.join(worker_folder,
                              common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, block_idx, 0))
    model = torch.load(model_path)
    model.load_state_dict(w_glob)

    return model


def worker(args):
    """
        The main function of the worker.
        `worker.py` loads a pretrained model, simplify it (one specific block), and short-term fine-tune the pruned model.
        Then, the accuracy and resource consumption of the simplified model will be recorded.
        `worker.py` finished with a finish file, which is utilized by `master.py`.
        
        Input: 
            args: command-line arguments
            
        raise:
            ValueError: when the num of block index >= simplifiable blocks (i.e. simplify nonexistent block or output layer)
    """

    # Set the GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Get the network utils.
    model = torch.load(args.model_path)
    
    network_utils = networkUtils.__dict__[args.arch](model, args.input_data_shape, args.dataset_path, args.finetune_lr)
    
    if need_simplify == 1:
        if network_utils.get_num_simplifiable_blocks() <= args.block:
            raise ValueError("Block index >= number of simplifiable blocks")
            
        network_def = network_utils.get_network_def_from_model(model)
        simplified_network_def, simplified_resource = (
            network_utils.simplify_network_def_based_on_constraint(network_def,
                                                                   args.block,
                                                                   args.constraint,
                                                                   args.resource_type,
                                                                   args.lookup_table_path))
        # Choose the filters.
        simplified_model = network_utils.simplify_model_based_on_network_def(simplified_network_def, model)
    
    else:
        simplified_model = model

    print('Original model:')
    print(model)
    print('')
    print('Simplified model:')
    print(simplified_model)


    devices = []

    save_path = os.path.join(args.master_folder, 
                             common.MASTER_DATALOADER_FILENAME_TEMPLATE.format(args.dataset))

    data_loader = dataLoader.__dict__[args.dataset](args.dataset_path)
    data_loader.load(save_path)

    devices = data_loader.group_idxs[args.block % len(data_loader.group_idxs)]

    print("Device number: "+ str(len(devices)))


    worker_begin = datetime.datetime.now()

    # fl tune
    fl_iter_num = args.round_number
    fl_model = simplified_model
    best_acc = 0
    group_data_num = [data_loader.get_device_data_size(d) for d in devices]
    # TODO()
    global_val_loader = data_loader.get_all_validation_data_loader()
    #data_loader.validation_data_loader_of_devices(devices)

    for fl_iter in range(fl_iter_num):
        # random select a group since second iteration
        if fl_iter > 0:
            devices = data_loader.group_idxs[random.randint(0,len(data_loader.group_idxs)-1)]
            group_data_num = [data_loader.get_device_data_size(d) for d in devices]

        for i in range(len(devices)):
            print('Start device ', i)
            device_begin = datetime.datetime.now()
            device_model = copy.deepcopy(fl_model)
            #TODO(zhaoyx): modify the logic
            if args.arch in ['mobilenetfed', 'mobilenet_imagenet', 'celebanetfed', 'mobilenet_imagenet_dali']:

                train_loader = data_loader.training_data_loader(devices[i])
                num_classes = int(common.DATASET_CLASSES_PARAMS[args.dataset])
                fine_tuned_model = network_utils.fine_tune(device_model, args.short_term_fine_tune_iteration, train_loader, num_classes)

            else:
                fine_tuned_model = network_utils.fine_tune(device_model, args.short_term_fine_tune_iteration)

            # Save the results.
            torch.save(fine_tuned_model,
                       os.path.join(args.worker_folder,
                                    common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)))
            # with open(os.path.join(args.worker_folder,
            #                        common.WORKER_DEVICE_ACCURACY_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
            #           'w') as file_id:
            #     file_id.write(str(fine_tuned_accuracy))
            # with open(os.path.join(args.worker_folder,
            #                        common.WORKER_DEVICE_RESOURCE_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
            #           'w') as file_id:
            #     file_id.write(str(simplified_resource))
            # with open(os.path.join(args.worker_folder,
            #                        common.WORKER_DEVICE_FINISH_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
            #           'w') as file_id:
            #     file_id.write('finished.')

            # release GPU memory
            del fine_tuned_model
            device_end = datetime.datetime.now()
            print ('Device running time: {} seconds'.format((device_end - device_begin).seconds))
            print('End device ', i)


        del fl_model
        # model fusion
        fl_model = _model_fusion(args.worker_folder, args.netadapt_iteration, args.block, len(devices), group_data_num)
        # get val accuracy
        fl_acc = network_utils.evaluate(fl_model,global_val_loader)
        print ("fl acc = {}".format(str(fl_acc)))
        if fl_acc > best_acc:
            best_acc = fl_acc
            torch.save(fl_model, os.path.join(args.worker_folder,
                                 common.WORKER_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block)))

    with open(os.path.join(args.worker_folder, common.WORKER_ACCURACY_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block)),
              'w') as file_id:
        file_id.write(str(fl_acc))
    with open(os.path.join(args.worker_folder, common.WORKER_RESOURCE_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block)),
            'w') as file_id:
        file_id.write(str(simplified_resource))


    print('Remove temp files')
    for i in range(len(devices)):
        temp_model_path = os.path.join(args.worker_folder,
                                common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i))
        os.remove(temp_model_path)





    worker_end = datetime.datetime.now()
    print ('Worker running time: {} seconds'.format((worker_end - worker_begin).seconds))
    print ("End this Worker, best acc = {}".format(str(best_acc)))
    with open(os.path.join(args.worker_folder,
                               common.WORKER_FINISH_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block)),
                  'w') as file_id:
        file_id.write('finished.')
    return 

if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('master_folder', type=str, 
                            help='directory where to read master partition')
    arg_parser.add_argument('worker_folder', type=str, 
                            help='directory where model and logging information will be saved')
    arg_parser.add_argument('model_path', type=str, help='path to model which is to be simplified')
    arg_parser.add_argument('block', type=int, help='index of block to be simplified')
    arg_parser.add_argument('resource_type', type=str, help='FLOPS/WEIGHTS/LATENCY')
    arg_parser.add_argument('constraint', type=float, help='floating value specifying resource constraint')
    arg_parser.add_argument('netadapt_iteration', type=int, help='netadapt iteration')
    arg_parser.add_argument('short_term_fine_tune_iteration', type=float, help='number of iterations of fine-tuning after simplification')
    arg_parser.add_argument('gpu', type=str, help='index of gpu to run short-term fine-tuning')
    arg_parser.add_argument('lookup_table_path', type=str, default='', help='path to lookup table')
    arg_parser.add_argument('dataset_path', type=str, default='', help='path to dataset')
    arg_parser.add_argument('input_data_shape', nargs=3, default=[], type=int, help='input shape (for ImageNet: `3 224 224`)')
    arg_parser.add_argument('arch', default='alexnet',
                    choices=network_utils_all,
                    help='network_utils: ' +
                        ' | '.join(network_utils_all) +
                        ' (default: alexnet)')
    arg_parser.add_argument('finetune_lr', type=float, default=0.001, help='short-term fine-tune learning rate')
    arg_parser.add_argument('device_number', type=int, default=10, help='number of devices total')
    arg_parser.add_argument('group_number', type=int, default=3, help='Group number.')
    arg_parser.add_argument('round_number', type=int, default=10, help='Round number.')
    arg_parser.add_argument('dataset',  default='cifar10', 
                        choices=data_loader_all,
                        help='dataset: ' +
                        ' | '.join(data_loader_all) +
                        ' (default: cifar10). Defines which dataset is used. If you want to use your own dataset, please specify here.')

    arg_parser.add_argument('need_simplify', type=int, default=1, help='If need simplify the network or not.')
    args = arg_parser.parse_args()

    # Launch a worker.
    worker(args)
