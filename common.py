# Master-related filenames.
MASTER_MODEL_FILENAME_TEMPLATE = 'iter_{}_best_model.pth.tar'

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