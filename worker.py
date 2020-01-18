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
    for i in range(len(devices)):
        print('Start device ', i)
        device_begin = datetime.datetime.now()
        device_model = copy.deepcopy(simplified_model)
        #TODO(zhaoyx): modify the logic
        if args.arch == 'mobilenetfed':
            train_loader = data_loader.training_data_loader(devices[i])
            num_classes = int(common.DATASET_CLASSES_PARAMS[args.dataset])
            fine_tuned_model = network_utils.fine_tune(device_model, args.short_term_fine_tune_iteration, train_loader, num_classes)
            val_loader = data_loader.validation_data_loader(devices[i])
            fine_tuned_accuracy = network_utils.evaluate(fine_tuned_model,val_loader)

        else:
            fine_tuned_model = network_utils.fine_tune(device_model, args.short_term_fine_tune_iteration)
            fine_tuned_accuracy = network_utils.evaluate(fine_tuned_model)

        print('Accuracy after finetune:', fine_tuned_accuracy)
        # TODO(zhaoyx): measure/simulate latency for different devices.
        latency = abs(np.random.normal(1))

        # Save the results.
        torch.save(fine_tuned_model,
                   os.path.join(args.worker_folder,
                                common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)))
        with open(os.path.join(args.worker_folder,
                               common.WORKER_DEVICE_ACCURACY_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
                  'w') as file_id:
            file_id.write(str(fine_tuned_accuracy))
        with open(os.path.join(args.worker_folder,
                               common.WORKER_DEVICE_RESOURCE_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
                  'w') as file_id:
            file_id.write(str(simplified_resource))
            file_id.write("\n")
            file_id.write(str(latency))
        with open(os.path.join(args.worker_folder,
                               common.WORKER_DEVICE_FINISH_FILENAME_TEMPLATE.format(args.netadapt_iteration, args.block, i)),
                  'w') as file_id:
            file_id.write('finished.')

        # release GPU memory
        del fine_tuned_model
        device_end = datetime.datetime.now()
        print ('Device running time: {} seconds'.format((device_end - device_begin).seconds))
        print('End device ', i)

    # release GPU memory
    del simplified_model
    worker_end = datetime.datetime.now()
    print ('Worker running time: {} seconds'.format((worker_end - worker_begin).seconds))
    print ("End this Worker")
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
    arg_parser.add_argument('dataset',  default='cifar10', 
                        choices=data_loader_all,
                        help='dataset: ' +
                        ' | '.join(data_loader_all) +
                        ' (default: cifar10). Defines which dataset is used. If you want to use your own dataset, please specify here.')
    

    args = arg_parser.parse_args()

    # Launch a worker.
    worker(args)
