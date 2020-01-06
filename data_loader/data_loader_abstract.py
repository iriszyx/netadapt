from abc import ABC, abstractmethod


class DataLoaderAbstract(ABC):

    def __init__(self):
        super().__init__()


    @abstractmethod
    def generate_device_data(self, device_number, is_iid):
        '''
            Split and sample data for all available devices.
            For datasets without user information, split dataset to device_number parts with IID/non-IID setting.
            For datasets with 1 < users number < device_number, resample some users.
            For datasets with users number >= device_number, sample device_number from users.

            Input: 
                `device_number`: total available devices number in the system
                `is_iid`: (bool) whether data in differet devices is IID or non-IID
                
            Output:
                `device_dataset_idx`: array which stores the idx of dataset
            
        '''
        pass


    @abstractmethod
    def generate_group_based_on_device_number(self, group_number):
        '''
            Generate group of devices with approximate total device number.
            
            Input:
                `group_number`: (int) group number 

            Output:
            
        '''
        pass


    @abstractmethod
    def generate_group_based_on_device_data_size(self, group_number):
        '''
            Generate group of devices with approximate total data size.
            
            Input:
                `group_number`: (int) group number 

            Output:
            
        '''
        pass


    @abstractmethod
    def training_data_loader(self, device_idx):
        '''
            Return training data loader for specified device with device_idx.
            
            Input:
                `device_idx`: (int) index of the device

            Output:
                `training_data_loader`: training data loader for the device 
            
        '''
        pass


    @abstractmethod
    def validation_data_loader(self, device_idx):
        '''
            Return validatio data loader for specified device with device_idx.
            
            Input:
                `device_idx`: (int) index of the device

            Output:
                `validation_data_loader`: validation data loader for the device 
            
        '''
        pass


    @abstractmethod
    def dump(self):
        '''
            Dump data loader to file.
            
        '''
        pass


    @abstractmethod
    def load(self):
        '''
            Load data loader from file.
            
        '''
        pass


    @abstractmethod
    def get_all_train_data_loader(self):

        pass


    @abstractmethod
    def get_all_validation_data_loader(self):

        pass
        

    @abstractmethod
    def get_test_data_loader(self):

        pass

    @abstractmethod
    def get_device_data_size(self):

        pass



