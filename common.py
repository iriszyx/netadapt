# Master-related filenames.
MASTER_MODEL_FILENAME_TEMPLATE = 'iter_{}_best_model.pth.tar'
MASTER_DATALOADER_FILENAME_TEMPLATE = 'data_loader_{}.txt'
MASTER_DATASET_SPLIT_FILENAME_TEMPLATE = 'dataset_split_index.txt'
MASTER_GROUP_FILENAME_TEMPLATE = 'group.txt'

# Worker-related filenames.
WORKER_MODEL_FILENAME_TEMPLATE = 'iter_{}_block_{}_model.pth.tar'
WORKER_ACCURACY_FILENAME_TEMPLATE = 'iter_{}_block_{}_accuracy.txt'
WORKER_RESOURCE_FILENAME_TEMPLATE = 'iter_{}_block_{}_resource.txt'
WORKER_LOG_FILENAME_TEMPLATE = 'iter_{}_block_{}_log.txt'
WORKER_FINISH_FILENAME_TEMPLATE = 'iter_{}_block_{}_finish.signal'

# Worker-related filenames for different devices.
WORKER_DEVICE_MODEL_FILENAME_TEMPLATE = 'iter_{}_block_{}_device_{}_model.pth.tar'
WORKER_DEVICE_ACCURACY_FILENAME_TEMPLATE = 'iter_{}_block_{}_device_{}_accuracy.txt'
WORKER_DEVICE_RESOURCE_FILENAME_TEMPLATE = 'iter_{}_block_{}_device_{}_resource.txt'
WORKER_DEVICE_LOG_FILENAME_TEMPLATE = 'iter_{}_block_{}_device_{}_log.txt'
WORKER_DEVICE_FINISH_FILENAME_TEMPLATE = 'iter_{}_block_{}_device_{}_finish.signal'

#dict
DATASET_CLASSES_PARAMS = {
    'cifar10': 10,
    'feminst': 62,
}