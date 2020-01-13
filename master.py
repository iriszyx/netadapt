from argparse import ArgumentParser
import os
import pickle
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
import time


'''
    The main file of NetAdapt.
    
    Launch workers to simplify and finetune pretrained models.
'''

# Define constants.
_MASTER_FOLDER_FILENAME = 'master'
_WORKER_FOLDER_FILENAME = 'worker'
_WORKER_PY_FILENAME = 'worker.py'
_HISTORY_PICKLE_FILENAME = 'history.pickle'
_HISTORY_TEXT_FILENAME = 'history.txt'
_SLEEP_TIME = 1

# Define keys.
_KEY_MASTER_ARGS = 'master_args'
_KEY_HISTORY = 'history'
_KEY_RESOURCE = 'resource'
_KEY_ACCURACY = 'accuracy'
_KEY_TIME_PHY = 'time_phy'
_KEY_TIME_SIM = 'time_sim'
_KEY_SOURCE_MODEL_PATH = 'source_model_path'
_KEY_BLOCK = 'block'
_KEY_ITERATION = 'iteration'
_KEY_GPU = 'gpu'
_KEY_MODEL = 'model'
_KEY_NETWORK_DEF = 'network_def'
_KEY_NUM_OUT_CHANNELS = 'num_out_channels'

# Supported network_utils
network_utils_all = sorted(name for name in networkUtils.__dict__
    if name.islower() and not name.startswith("__")
    and callable(networkUtils.__dict__[name]))

# Supported data_loaders
data_loader_all = sorted(name for name in dataLoader.__dict__
    if name.islower() and not name.startswith("__")
    and callable(dataLoader.__dict__[name]))


def _launch_worker(master_folder,
                   worker_folder, model_path, block, resource_type, constraint, netadapt_iteration,
                   short_term_fine_tune_iteration, input_data_shape, job_list, available_gpus, 
                   lookup_table_path, dataset_path, model_arch):
    '''
        `master.py` launches several `worker.py`.
        Each `worker.py` prunes one specific block and fine-tune it.
        This function launches one worker to run on one gpu.
        
        Input:
            `worker_folder`: (string) directory where `worker.py` will save models.
            `model_path`:(string) path to model which `worker.py` will load as pretrained model.
            `block`: (int) index of block to be simplified.
            `resource_type`: (string) (e.g. `WEIGHTS`, `FLOPS`, and `LATENCY`).
            `constraint`: (float) the value of constraints (e.g. 10**6 (weights)).
            `netadapt_iteration`: (int) indicates the current iteration of NetAdapt.
            `short_term_fine_tune_iteration`: (int) short-term fine-tune iteration.
            `input_data_shape`: (list) input data shape (C, H, W).
            `job_list`: (list of dict) list of current jobs. Each job is a dict, showing current iteration, block and gpu idx.
            `available_gpus`: (list) list of available gpu idx.
            `lookup_table_path`: (string) path to lookup table.
            `dataset_path`: (string) path to dataset.
            `model_arch`: (string) specifies which network_utils will be used.
        
        Output:
            updated_job_list: (list of dict)
            updated_available_gpus: (list)
    '''
    updated_job_list = job_list.copy()
    updated_available_gpus = available_gpus.copy()
    gpu = updated_available_gpus[0]

    if lookup_table_path == None:
        lookup_table_path = ''

    print('  Launch a worker for block {}'.format(block))
    with open(os.path.join(worker_folder,
                           common.WORKER_LOG_FILENAME_TEMPLATE.format(netadapt_iteration, block)), 'w') as file_id:
        command_list = [sys.executable, _WORKER_PY_FILENAME, master_folder, worker_folder, model_path, str(block), resource_type,
                        str(constraint), str(netadapt_iteration), str(short_term_fine_tune_iteration), str(gpu),
                        lookup_table_path, dataset_path] + [str(e) for e in input_data_shape] + [model_arch] +\
                        [str(args.finetune_lr)] +\
                        [str(args.device_number)] + [str(args.group_number)] + [str(args.dataset)]


        
        print(command_list)
        
        subprocess.Popen(command_list, stdout=file_id, stderr=file_id)

    updated_job_list.append({_KEY_ITERATION: netadapt_iteration, _KEY_BLOCK: block, _KEY_GPU: gpu})
    del updated_available_gpus[0]

    return updated_job_list, updated_available_gpus


def _update_job_list_and_available_gpus(worker_folder, job_list, available_gpus):
    '''
        update job list and available gpu list based on whether a worker finishes pruning and fine-tuning.
        
        Input:
            `worker_folder`: (string) directory where `worker.py` will save models.
            `job_list`: (list of dict) list of current jobs. Each job is a dict, showing current iteration, block and gpu idx.
            `available_gpus`: (list) list of available gpu idx.
        
        Output:
            `updated_job_list`: (list of dict) if a worker finishes its job, the job will be removed from this list.
            `updated_available_gpus`: (list) if a worker finishes its job, the gpu will be available.
    '''
    updated_job_list = []
    updated_available_gpus = available_gpus.copy()
    for job in job_list:
        if os.path.exists(os.path.join(worker_folder, common.WORKER_FINISH_FILENAME_TEMPLATE.format(job[_KEY_ITERATION],
                                                                                                    job[_KEY_BLOCK]))):
            # Find corresponding finish file of worker
            updated_available_gpus.append(job[_KEY_GPU])
        else:
            updated_job_list.append(job)

    return updated_job_list, updated_available_gpus


def _find_best_model(worker_folder, iteration, num_blocks, starting_accuracy, starting_resource):
    '''
        After all workers finish jobs, select the model with best accuracy-to-resource ratio
        
        Input:
            `worker_folder`: (string) directory where `worker.py` will save models.
            `iteration`: (int) NetAdapt iteration.
            `num_blocks`: (int) num of simplifiable blocks at each iteration.
            `starting_accuracy`: (float) initial accuracy before pruning and fine-tuning.
            `start_resource`: (float) initial resource sonsumption.
        
        Output:
            `best_accuracy`: (float) accuracy of the best pruned model.
            `best_model_path`: (string) path to the best model.
            `best_resource`: (float) resource consumption of the best model.
            `best_block`: (int) block index of the best model.
    '''
    
    best_ratio = float('Inf')
    best_accuracy = 0.0
    best_model_path = None
    best_resource = None
    best_block = None
    for block_idx in range(num_blocks):
        with open(os.path.join(worker_folder, common.WORKER_ACCURACY_FILENAME_TEMPLATE.format(iteration, block_idx)),
                  'r') as file_id:
            accuracy = float(file_id.read())
        with open(os.path.join(worker_folder, common.WORKER_RESOURCE_FILENAME_TEMPLATE.format(iteration, block_idx)),
                  'r') as file_id:
            resource = float(file_id.read())
        #ratio_resource_accuracy = (starting_accuracy - accuracy) / (starting_resource - resource + 1e-5)
        ratio_resource_accuracy = (starting_accuracy - accuracy + 1e-6) / (starting_resource - resource + 1e-5)
        
        print('Block id {}: resource {}, accuracy {}'.format(block_idx, resource, accuracy))
        if resource < starting_resource and ratio_resource_accuracy < best_ratio:
        #if resource < starting_resource and accuracy > best_accuracy:
            best_ratio = ratio_resource_accuracy
            best_accuracy = accuracy
            best_model_path = os.path.join(worker_folder,
                                           common.WORKER_MODEL_FILENAME_TEMPLATE.format(iteration, block_idx))
            best_resource = resource
            best_block = block_idx
    print('Best block id: {}\n'.format(best_block))

    return best_accuracy, best_model_path, best_resource, best_block


def _find_best_network_with_metric_fusion(worker_folder, iteration, num_blocks, starting_accuracy, starting_resource,
                                            group_len, group_val_data_num):
    '''
        After all workers finish jobs, select the network with best accuracy-to-resource ratio after metric fusion.
        
        Input:
            `worker_folder`: (string) directory where `worker.py` will save models.
            `iteration`: (int) NetAdapt iteration.
            `num_blocks`: (int) num of simplifiable blocks at each iteration.
            `starting_accuracy`: (float) initial accuracy before pruning and fine-tuning.
            `start_resource`: (float) initial resource sonsumption.
            `group_len`: (list[int]) num of workers of every group.
        
        Output:
            `best_accuracy`: (float) accuracy of the best pruned network.
            `best_model_path`: (string) path to the best model.
            `best_resource`: (float) resource consumption of the best network.
            `best_block`: (int) block index of the best network.
    '''
    
    best_ratio = float('Inf')
    best_accuracy = 0.0
    best_model_path = None
    best_resource = None
    best_block = None
    for block_idx in range(num_blocks):
        vaild_device_num = 0.0
        vaild_device_size = 0.0
        accuracy_sum = 0.0
        resource_sum = 0.0
        worker_num = group_len[block_idx%len(group_len)]
        for worker_idx in range(worker_num):
            with open(os.path.join(worker_folder, common.WORKER_DEVICE_ACCURACY_FILENAME_TEMPLATE.format(iteration, block_idx,
                      worker_idx)), 'r') as file_id:
                accuracy = float(file_id.read())
            os.remove(os.path.join(worker_folder, common.WORKER_DEVICE_ACCURACY_FILENAME_TEMPLATE.format(iteration, block_idx,
                      worker_idx)))
            with open(os.path.join(worker_folder, common.WORKER_DEVICE_RESOURCE_FILENAME_TEMPLATE.format(iteration, block_idx,
                      worker_idx )), 'r') as file_id:
                resource = float(file_id.readline())
            os.remove(os.path.join(worker_folder, common.WORKER_DEVICE_RESOURCE_FILENAME_TEMPLATE.format(iteration, block_idx,
                      worker_idx)))
            os.remove(os.path.join(worker_folder, common.WORKER_DEVICE_FINISH_FILENAME_TEMPLATE.format(iteration, block_idx,
                      worker_idx)))
            #TODO(zhaoyx): fix bug
            if accuracy < 0:
                print('skip')
                print('Block id {}: resource {}, accuracy {}'.format(block_idx, resource, accuracy))
                continue
            # Metric fusion group_val_data_num[block_idx % len(group_len)][worker_idx]
            if resource < starting_resource:
                vaild_device_num += 1
                vaild_device_size += group_val_data_num[block_idx % len(group_len)][worker_idx]
                accuracy_sum += accuracy * group_val_data_num[block_idx % len(group_len)][worker_idx]
                resource_sum += resource
        
        if vaild_device_num < 1:
            continue
        
        accuracy = accuracy_sum / vaild_device_size
        resource = resource_sum / vaild_device_num

        #ratio_resource_accuracy = (starting_accuracy - accuracy) / (starting_resource - resource + 1e-5)
        ratio_resource_accuracy = (starting_accuracy - accuracy + 1e-6) / (starting_resource - resource + 1e-5)
        
        print('Block id {}: resource {}, accuracy {}'.format(block_idx, resource, accuracy))
        if resource < starting_resource and ratio_resource_accuracy < best_ratio:
        #if resource < starting_resource and accuracy > best_accuracy:
            best_ratio = ratio_resource_accuracy
            best_accuracy = accuracy
            # best_model_path = os.path.join(worker_folder,
            #                                common.WORKER_MODEL_FILENAME_TEMPLATE.format(iteration, block_idx))
            best_resource = resource
            best_block = block_idx
    print('Best block id: {}\n'.format(best_block))

    return best_accuracy, best_resource, best_block

# def _fed_avg_old(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg

def _fed_avg(w, data_num):
    w_avg = {k: [torch.mul(w[i][k], data_num[i]) for i in range(len(w))] for k in w[0]}
    w_avg = {k: functools.reduce(lambda x,y: x + y, w_avg[k]) / np.sum(data_num) for k in w_avg}
    return w_avg

# def need_fl_tune(iter_id, total_iter_num):
#     '''
#         Shall we run federated learning on the fused model (long-term fine-tune)?
#     '''
#     return iter_id == (total_iter_num - 1)

# def run_fl(model_path, data_loader, network_utils, skip_ratio=0.0):
#     '''
#         Run federated learning
#         Input:
#             'model_path': fused model's path, waiting to be fl-ed
#             'data_loader': data loader for devices
#         Output:
#             'fl_model_path': a new model path where fl-ed model is stored
#     '''
#     return model_path

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
    best_model_path = os.path.join(worker_folder,
                                   common.WORKER_MODEL_FILENAME_TEMPLATE.format(iteration, block_idx))
    w_array = []

    for device_idx in range(device_num):
        model_path = os.path.join(worker_folder,
                                  common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(iteration, block_idx, device_idx))
        model = torch.load(model_path)
        s1 = model.state_dict()
        w_array.append(copy.deepcopy(s1))
        del model
        del s1

    w_glob = _fed_avg(w_array, data_num)

    model_path = os.path.join(worker_folder,
                              common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(iteration, block_idx, 0))
    model = torch.load(model_path)
    model.load_state_dict(w_glob)

    torch.save(model, best_model_path)
    return best_model_path


def _save_and_print_history(network_utils, history, pickle_file_path, text_file_path):
    '''
        save history info (log: history.txt, history file: history.pickle)
        
        Input:
            `network_utils`: (defined in network_utils/network_utils_*) use the .extra_history_info()
                            to get the num of output channels.
            `history`: (dict) records accuracy, resource, block idx, model path for each iteration and 
                     input arguments.
            `pickle_file_path`: (string) path to save history dict.
            `text_file_path`: (string) path to save history log.
    '''
    with open(pickle_file_path, 'wb') as file_id:
        pickle.dump(history, file_id)
    with open(text_file_path, 'w') as file_id:
        file_id.write('Iteration,Accuracy,PhyTime,SimTime,Resource,Block,Source Model\n')
        for iter in range(len(history[_KEY_HISTORY])):
            
            # assume the extra hisotry info is the # of output channels per layer
            num_filters_str = network_utils.extra_history_info(history[_KEY_HISTORY][iter][_KEY_NETWORK_DEF])
            file_id.write('{},{},{},{},{},{},{},{}\n'.format(iter, history[_KEY_HISTORY][iter][_KEY_ACCURACY],
                                                       history[_KEY_HISTORY][iter][_KEY_TIME_PHY],
                                                       history[_KEY_HISTORY][iter][_KEY_TIME_SIM],
                                                       history[_KEY_HISTORY][iter][_KEY_RESOURCE],
                                                       history[_KEY_HISTORY][iter][_KEY_BLOCK],
                                                       history[_KEY_HISTORY][iter][_KEY_SOURCE_MODEL_PATH],
                                                       num_filters_str))


def master(args):
    """
        The main function of the master.

        Note: iteration 0 means the initial model.
        
        Input: 
            args: input arguments 
            
        raise:
            ValueError: when:
                (1) no available gpus (i.e. len(args.gpus) == 0)
                (2) resume from previous iteration and required to use lookup table but no loookup table found
                (3) files exist under working_folder/master or working_folder/worker and not use `--resume`
                (4) target resource is not achievable (i.e. the resource consumption at a certain iteration is the same as that at previous iteration)
    """

    # Set the important paths.
    master_folder = os.path.join(args.working_folder, _MASTER_FOLDER_FILENAME)
    worker_folder = os.path.join(args.working_folder, _WORKER_FOLDER_FILENAME)
    history_pickle_file = os.path.join(master_folder, _HISTORY_PICKLE_FILENAME)
    history_text_file = os.path.join(master_folder, _HISTORY_TEXT_FILENAME)


    # Get available GPUs.
    available_gpus = args.gpus
    if len(available_gpus) == 0:
        raise ValueError('At least one gpu must be specified.')

    # Resume or do iteration 0.
    if args.resume:
        old_args = copy.deepcopy(args)
        with open(history_pickle_file, 'rb') as file_id:
            history = pickle.load(file_id)
        args = history[_KEY_MASTER_ARGS]
        args.max_iters = old_args.max_iters

        # Initialize variables.
        current_iter = len(history[_KEY_HISTORY]) - 1
        current_resource = history[_KEY_HISTORY][-1][_KEY_RESOURCE]
        current_model_path = os.path.join(master_folder,
                                          common.MASTER_MODEL_FILENAME_TEMPLATE.format(current_iter))
        current_accuracy = history[_KEY_HISTORY][-1][_KEY_ACCURACY]

        # Get the network utils.
        model = torch.load(current_model_path, map_location=lambda storage, loc: storage)
             
        # Select network_utils.
        model_arch = args.arch
        network_utils = networkUtils.__dict__[model_arch](model, args.input_data_shape, args.dataset_path)

        if args.lookup_table_path != None and not os.path.exists(args.lookup_table_path):
            errMsg = 'Resume from a previous task but the {} lookup table is not found.'.format(args.resource_type)
            raise ValueError(errMsg)
            del model

        # Print the message.
        print(('Resume from iteration {:>3}: current_accuracy = {:>8.3f}, '
               'current_resource = {:>8.3f}').format(current_iter, current_accuracy, current_resource))
        print('arguments:', args)
        
    else:
        # Initialize the iteration.
        current_iter = 0

        # Create the folder structure.
        if not os.path.exists(args.working_folder):
            os.makedirs(args.working_folder)
            print('Create directory', args.working_folder)
        if not os.path.exists(master_folder):
            os.mkdir(master_folder)
            print('Create directory', master_folder)
        elif os.listdir(master_folder):
            errMsg = 'Find previous files in the master directory {}. Please use `--resume` or delete those files'.format(master_folder)
            raise ValueError(errMsg)
            
        if not os.path.exists(worker_folder):
            os.mkdir(worker_folder)
            print('Create directory', worker_folder)
        elif os.listdir(worker_folder):
            errMsg = 'Find previous files in the worker directory {}. Please use `--resume` or delete those files'.format(worker_folder)
            raise ValueError(errMsg) 
           
        # Backup the initial model.
        current_model_path = os.path.join(master_folder,
                                          common.MASTER_MODEL_FILENAME_TEMPLATE.format(current_iter))
        copyfile(args.init_model_path, current_model_path)

        # Initialize variables.
        model = torch.load(current_model_path)
        
        # Select network_utils.
        model_arch = args.arch
        network_utils = networkUtils.__dict__[model_arch](model, args.input_data_shape, args.dataset_path)

        network_def = network_utils.get_network_def_from_model(model)
        if args.lookup_table_path != None and not os.path.exists(args.lookup_table_path):
            warnMsg = 'The {} lookup table is not found and going to be built.'.format(args.resource_type)
            warnings.warn(warnMsg)
            network_utils.build_lookup_table(network_def, args.resource_type, args.lookup_table_path)
        current_resource = network_utils.compute_resource(network_def, args.resource_type, args.lookup_table_path)

        # Dataset loading and partition.
        data_loader = dataLoader.__dict__[args.dataset](args.dataset_path)
        val_loader = data_loader.get_all_validation_data_loader()
        current_accuracy = network_utils.evaluate(model, val_loader)
        current_block = None
        
        if args.init_resource_reduction == None:
            args.init_resource_reduction = args.init_resource_reduction_ratio*current_resource
            print('`--init_resource_reduction` is not specified')
            print('Use `--init_resource_reduction_ratio` ({}) to get `init_resource_reduction` ({})\n'.format(
                    args.init_resource_reduction_ratio, args.init_resource_reduction))
        if args.budget == None:
            args.budget = args.budget_ratio*current_resource
            print('`--budget` is not specified')
            print('Use `--budget_ratio` ({}) to get `budget` ({})\n'.format(
                    args.budget_ratio, args.budget))

        # Create and save the history.
        history = {_KEY_MASTER_ARGS: args, _KEY_HISTORY: []}
        history[_KEY_HISTORY].append({_KEY_RESOURCE: current_resource,
                                      _KEY_SOURCE_MODEL_PATH: args.init_model_path,
                                      _KEY_ACCURACY: current_accuracy,
                                      _KEY_TIME_PHY: -1,
                                      _KEY_TIME_SIM: -1,
                                      _KEY_BLOCK: current_block,
                                      _KEY_NETWORK_DEF: network_def})
        _save_and_print_history(network_utils, history, history_pickle_file, history_text_file)
        del model, network_def

        # Print the message.
        print(('Start from iteration {:>3}: current_accuracy = {:>8.3f}, '
               'current_resource = {:>8.3f}').format(current_iter, current_accuracy, current_resource))
    
    # Dataset loading and partition.
    data_loader = dataLoader.__dict__[args.dataset](args.dataset_path)
    is_iid = True if args.dataset == 'cifar10' else False #TODO: in a smarter way
    device_data_idxs = data_loader.generate_device_data(args.device_number,is_iid)
    group_idxs = data_loader.generate_group_based_on_device_data_size(args.group_number)
    group_len = [len(group_idxs[i]) for i in range(len(group_idxs))]

    # print(device_data_idxs)
    # print(group_idxs)


    save_path = os.path.join(master_folder, 
                             common.MASTER_DATALOADER_FILENAME_TEMPLATE.format(args.dataset))
    data_loader.dump(save_path)    
                
    init_resource = current_resource

    current_iter += 1


    # Start adaptation.
    while current_iter <= args.max_iters and current_resource > args.budget:
        print ('Iter: {}, Resource: {}, Resource_budget: {}'.format(current_iter, current_resource, args.budget))
        iter_start_time = datetime.datetime.now()
        
        # Set the target resource.
        target_resource = current_resource - args.init_resource_reduction * (
                args.resource_reduction_decay ** (current_iter - 1))

        # Print the message.
        print('===================================================================')
        print(
            ('Process iteration {:>3}: current_accuracy = {:>8.3f}, '
             'current_resource = {:>8.3f}, target_resource = {:>8.3f}').format(
                current_iter, current_accuracy, current_resource, target_resource))

        # Launch the workers.
        job_list = []
        
        # Launch worker for each block
        for block_idx in range(network_utils.get_num_simplifiable_blocks()):
            # Check and update the gpu availability.
            job_list, available_gpus = _update_job_list_and_available_gpus(worker_folder, job_list, available_gpus)
            while not available_gpus:
                # print('  Wait for the next available gpu...')
                time.sleep(_SLEEP_TIME)
                job_list, available_gpus = _update_job_list_and_available_gpus(worker_folder, job_list, available_gpus)

            # Launch a worker.
            job_list, available_gpus = _launch_worker(master_folder,
                                                      worker_folder, current_model_path, block_idx, args.resource_type,
                                                      target_resource, current_iter,
                                                      args.short_term_fine_tune_iteration, args.input_data_shape,
                                                      job_list, available_gpus, args.lookup_table_path,
                                                      args.dataset_path, args.arch)
            print('Update job list:     ', job_list)
            print('Update available gpu:', available_gpus, '\n')

        # Wait until all the workers finish.
        job_list, available_gpus = _update_job_list_and_available_gpus(worker_folder, job_list, available_gpus)
        while job_list:
            time.sleep(_SLEEP_TIME)
            job_list, available_gpus = _update_job_list_and_available_gpus(worker_folder, job_list, available_gpus)

        # # Find the best model.
        # best_accuracy, best_model_path, best_resource, best_block = (
        #     _find_best_model(worker_folder, current_iter, network_utils.get_num_simplifiable_blocks(), current_accuracy,
        #                      current_resource))

        # Find best model by metric fusion
        group_val_data_num = [[data_loader.get_device_val_data_size(d) for d in group_idxs[group_id]] for group_id in range(args.group_number)]
        best_accuracy, best_resource, best_block = (
            _find_best_network_with_metric_fusion(worker_folder, current_iter, network_utils.get_num_simplifiable_blocks(), 
                                                  current_accuracy, current_resource, group_len, group_val_data_num))
        # Model fusion, and generate best model at best model path
        group_id = best_block % len(group_len)
        device_num =  group_len[group_id]
        group_data_num = [data_loader.get_device_data_size(d) for d in group_idxs[group_id]]
        print ('Group: {}, group device data: {}'.format(group_id, group_data_num))
        best_model_path = _model_fusion(worker_folder, current_iter, best_block, device_num, group_data_num)

        # Check if we need a long-term fine-tune (federated learning)
        # if need_fl_tune(current_iter, args.max_iters):
        #     best_model_path = run_fl(best_model_path, data_loader, network_utils)

        # Check whether the target_resource is achieved.
        if not best_model_path:
            raise ValueError('target_resource {} is not achievable in iter {}.'.format(target_resource, current_iter))
        if best_resource > target_resource:
            warnMsg = "Iteration {}: target resource {} is not achieved. Current best resource is {}".format(current_iter, target_resource, best_resource)
            warnings.warn(warnMsg)
        
        # Update the variables.
        current_model_path = os.path.join(master_folder,
                                          common.MASTER_MODEL_FILENAME_TEMPLATE.format(current_iter))
        copyfile(best_model_path, current_model_path)
        #os.remove(best_model_path)
        current_accuracy = best_accuracy
        current_resource = best_resource
        current_block = best_block
        
        if args.save_interval == -1 or (current_iter % args.save_interval != 0):
            for block_idx in range(network_utils.get_num_simplifiable_blocks()):
                device_num = group_len[block_idx % len(group_len)]
                for worker_idx in range(device_num):
                    temp_model_path = os.path.join(worker_folder, common.WORKER_DEVICE_MODEL_FILENAME_TEMPLATE.format(current_iter, block_idx, worker_idx))
                    os.remove(temp_model_path)
                    print('Remove', temp_model_path)
                print(' ')

        # the running time for this iteration (simulation)
        model_size = os.path.getsize(current_model_path)
        sim_iter_time = fns.get_fl_st_iter_time(
            args.arch, model_size, network_utils.get_num_simplifiable_blocks(),
            args.group_number, network_utils.batch_size * args.short_term_fine_tune_iteration,
            current_resource / init_resource)
        print ('Running time for one iteration: ' + str(sim_iter_time) + ' seconds')

        # the actual running time (physical)
        phy_iter_time = (datetime.datetime.now() - iter_start_time).seconds

        # Save and print the history.
        model = torch.load(current_model_path)
        if type(model) is dict:
            model = model[_KEY_MODEL]
        network_def = network_utils.get_network_def_from_model(model)
        history[_KEY_HISTORY].append({_KEY_RESOURCE: current_resource,
                                      _KEY_SOURCE_MODEL_PATH: best_model_path,
                                      _KEY_TIME_PHY: phy_iter_time,
                                      _KEY_TIME_SIM: sim_iter_time,
                                      _KEY_ACCURACY: current_accuracy,
                                      _KEY_BLOCK: current_block,
                                      _KEY_NETWORK_DEF: network_def})
        _save_and_print_history(network_utils, history, history_pickle_file, history_text_file)
        del model, network_def

        current_iter += 1
        
        print('Finish iteration {}: time {}'.format(current_iter-1, phy_iter_time))


if __name__ == '__main__':
    # Parse the input arguments.
    arg_parser = ArgumentParser()
    arg_parser.add_argument('working_folder', type=str, 
                            help='Root folder where models, related files and history information are saved.')
    arg_parser.add_argument('input_data_shape', nargs=3, default=[3, 224, 224], type=int,
                            help='Input data shape (C, H, W) (default: 3 224 224).')
    arg_parser.add_argument('-gp', '--gpus', nargs='+', default=[0], type=int,
                            help='Indices of available gpus (default: 0).')
    arg_parser.add_argument('-re', '--resume', action='store_true',
                            help='Resume from previous iteration. In order to resume, specify `--resume` and specify `working_folder` as the one you want to resume.')
    arg_parser.add_argument('-im', '--init_model_path',
                            help='Path to pretrained model.')
    arg_parser.add_argument('-mi', '--max_iters', type=int, default=10,
                            help='Maximum iteration of removing filters and short-term fine-tune (default: 10).')
    arg_parser.add_argument('-lr', '--finetune_lr', type=float, default=0.001, 
                            help='Short-term fine-tune learning rate (default: 0.001).')
    
    arg_parser.add_argument('-bu', '--budget', type=float, default=None,
                            help='Resource constraint. If resource < `budget`, the process is terminated.')
    arg_parser.add_argument('-bur', '--budget_ratio', type=float, default=0.25,
                            help='If `--budget` is not specified, `buget` = `budget_ratio`*(pretrained model resource) (default: 0.25).')
    
    arg_parser.add_argument('-rt', '--resource_type', type=str, default='FLOPS', 
                            help='Resource constraint type (default: FLOPS). We currently support `FLOPS`, `WEIGHTS`, and `LATENCY` (device cuda:0). If you want to add other resource types, please modify network_util.')
    
    arg_parser.add_argument('-ir', '--init_resource_reduction', type=float, default=None, 
                            help='For each iteration, target resource = current resource - `init_resource_reduction`*(`resource_reduction_decay`**(iteration-1)).')
    arg_parser.add_argument('-irr', '--init_resource_reduction_ratio', type=float, default=0.025,
                            help='If `--init_resource_reduction` is not specified, `init_resource_reduction` = `init_resource_reduction_ratio`*(pretrained model resource) (default: 0.025).')
    
    
    arg_parser.add_argument('-rd', '--resource_reduction_decay', type=float, default=0.96,
                            help='For each iteration, target resource = current resource - `init_resource_reduction`*(`resource_reduction_decay`**(iteration-1)) (default: 0.96).')
    arg_parser.add_argument('-st', '--short_term_fine_tune_iteration', type=int, default=10, 
                            help='Short-term fine-tune iteration (default: 10).')
    
    arg_parser.add_argument('-lt', '--lookup_table_path', type=str, default=None, 
                            help='Path to lookup table.')
    arg_parser.add_argument('-dp', '--dataset_path', type=str, default='', 
                            help='Path to dataset.')
    
    arg_parser.add_argument('-a', '--arch', metavar='ARCH', default='alexnet',
                    choices=network_utils_all,
                    help='network_utils: ' +
                        ' | '.join(network_utils_all) +
                        ' (default: alexnet). Defines how networks are pruned, fine-tuned, and evaluated. If you want to use your own method, please specify here.')
    
    arg_parser.add_argument('-si', '--save_interval', type=int, default=-1,
                            help='Interval of iterations that all pruned models at the same iteration will be saved. Use `-1` to save only the best model at each iteration. Use `1` to save all models at each iteration. (default: -1).')
    
    # Added parameters.
    arg_parser.add_argument('-dn', '--device_number', type=int, default=3, 
                            help='Total device number.')
    arg_parser.add_argument('-d', '--dataset',  default='cifar10', 
                        choices=data_loader_all,
                        help='dataset: ' +
                        ' | '.join(data_loader_all) +
                        ' (default: cifar10). Defines which dataset is used. If you want to use your own dataset, please specify here.')
    arg_parser.add_argument('-gn', '--group_number', type=int, default=13, 
                            help='Group number.')

    print(network_utils_all)
    
    args = arg_parser.parse_args()

    # Launch the master.
    print(args)
    master(args)
