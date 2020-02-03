import numpy as np
import os
import scipy.misc
import time
import io
import torch.utils.data
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.datasets as datasets
from nvidia.dali.pipeline import Pipeline
import torchvision.transforms as transforms
from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator

def collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [np.array([item[1]], dtype = np.uint8) for item in batch]
    return images, labels

class ExternalSourceTrainPipeline(Pipeline):
    def __init__(self, source, batch_size, num_threads, device_id):
        super(ExternalSourceTrainPipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        dali_device = "gpu"
        self.source = source
        self.source_iter = iter(source)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        # print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return (output, self.labels)

    def iter_setup(self):
        try:
          p = self.source_iter.next()
        except:
          print("Exception occured")
          self.source_iter = iter(self.source)
          p = self.source_iter.next()
        images, labels = p
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def size(self):
        return len(self.source)


class ExternalSourceValPipeline(Pipeline):
    def __init__(self, source, batch_size, num_threads, device_id):
        super(ExternalSourceValPipeline, self).__init__(batch_size,
                                      num_threads,
                                      device_id,
                                      seed=12)
        dali_device = "gpu"
        self.source = source
        self.source_iter = iter(source)
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=256, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(224, 224),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

        #print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return (output, self.labels)

    def iter_setup(self):
        try:
          p = self.source_iter.next()
        except:
          print("Exception occured")
          self.source_iter = iter(self.source)
          p = self.source_iter.next()
        images, labels = p
        self.feed_input(self.jpegs, images)
        self.feed_input(self.labels, labels)

    def size(self):
        return len(self.source)


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, local_rank=0, world_size=1):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        dali_device = "gpu"
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=True)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=crop, random_area=[0.08, 1.25])
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images, mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, local_rank=0, world_size=1):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def get_imagenet_iter_dali(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                           world_size=1,
                           local_rank=0):
    if type == 'train':
        pip_train = HybridTrainPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                    data_dir=image_dir + '/train',
                                    crop=crop, world_size=world_size, local_rank=local_rank)
        pip_train.build()
        dali_iter_train = DALIClassificationIterator(pip_train, size=pip_train.epoch_size("Reader") // world_size)
        return dali_iter_train
    elif type == 'val':
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/val',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size)
        return dali_iter_val
    else:
        pip_val = HybridValPipe(batch_size=batch_size, num_threads=num_threads, device_id=local_rank,
                                data_dir=image_dir + '/train',
                                crop=crop, size=val_size, world_size=world_size, local_rank=local_rank)
        pip_val.build()
        dali_iter_val = DALIClassificationIterator(pip_val, size=pip_val.epoch_size("Reader") // world_size)
        return dali_iter_val

def get_imagenet_iter_torch(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    if type == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(crop, scale=(0.08, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(val_size),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_threads,
                                                 pin_memory=True)
    return dataloader

def get_imagenet_iter_dali_with_custom_dataloader(dataset_loader, batch_size, num_threads):
    
    pipe = ExternalSourceTrainPipeline(source=dataset_loader, batch_size=batch_size, num_threads=num_threads, device_id=0)
    pipe.build()

    dali_iter = DALIGenericIterator([pipe], ['data', 'label'], size = pipe.size()*batch_size)
    return dali_iter

def get_imagenet_iter_dali_with_custom_dataset(type, image_dir, batch_size, num_threads, device_id, num_gpus, crop, val_size=256,
                            world_size=1, local_rank=0):
    def _tr(im_resize):
        im_resize = im_resize.convert('RGB')
        buf = io.BytesIO()
        im_resize.save(buf, format='JPEG')
        ret = np.frombuffer(buf.getvalue(), dtype = np.uint8)
        return ret
    
    transform = transforms.Compose([
        transforms.Lambda(lambda x: _tr(x)),
    ])

    if type == 'train':
        dataset = datasets.ImageFolder(image_dir + '/train', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True, collate_fn=collate_fn)#, drop_last=True)
        pipe = ExternalSourceTrainPipeline(source=dataloader, batch_size=batch_size, num_threads=num_threads, device_id=0)
    

    else:
        dataset = datasets.ImageFolder(image_dir + '/val', transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_threads,
                                                 pin_memory=True, collate_fn=collate_fn)#, drop_last=True)
        pipe = ExternalSourceValPipeline(source=dataloader, batch_size=batch_size, num_threads=num_threads, device_id=0)
    

    pipe.build()
    
    dali_iter = DALIGenericIterator([pipe], ['data', 'label'], size = pipe.size()*batch_size)
    return dali_iter


if __name__ == '__main__':
    # train_loader = get_imagenet_iter_dali(type='train', image_dir='/data1/dataset', batch_size=256,
    #                                       num_threads=16, crop=224, device_id=0, num_gpus=1)
    # print('start iterate')
    # start = time.time()
    # num_iter = 0
    # for i, data in enumerate(train_loader):
    #     print(data)
    #     images = data[0]["data"].cuda(non_blocking=True)
    #     labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
    #     num_iter += 1
    #     if num_iter == 1:
    #         break
    # end = time.time()
    # print('end iterate')
    # print(num_iter)
    # print('dali iterate time: %fs' % (end - start))


   
    train_loader = get_imagenet_iter_dali_with_custom_dataset(type='train', image_dir='/data1/dataset', batch_size=256,
                                           num_threads=16, crop=224, device_id=0, num_gpus=1)

    print('start iterate')
    start = time.time()
    num_iter = 0
    for i, data in enumerate(train_loader):
        print(data)
        images = data[0]["data"].cuda(non_blocking=True)
        labels = data[0]["label"].squeeze().long().cuda(non_blocking=True)
        print(labels)
        labels = data[0]["label"].cuda(non_blocking=True)
        print(labels)
        num_iter+=1
        if num_iter == 1:
            break
    end = time.time()
    print('end iterate')
    print(num_iter)
    print('dali iterate time: %fs' % (end - start))

    # train_loader = get_imagenet_iter_torch(type='train', image_dir='/data1/dataset', batch_size=256,
    #                                        num_threads=16, crop=224, device_id=0, num_gpus=1)
    # print('start iterate')
    # start = time.time()
    # num_iter = 0
    # for i, data in enumerate(train_loader):
    #     # images = data[0].cuda(non_blocking=True)
    #     # labels = data[1].cuda(non_blocking=True)
    #     images = data[0].cuda()
    #     labels = data[1].cuda()
    #     num_iter += 1
    #     if num_iter == 100:
    #         break
    # end = time.time()
    # print('end iterate')
    # print(num_iter)
    # print('torch iterate time: %fs' % (end - start))