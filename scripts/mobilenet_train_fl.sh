CUDA_VISIBLE_DEVICES=0 python train_fl.py models/mobilenet/imagenet_netadapt ALL /data1/dataset \
    -ge 3 -le 2 -lr 0.045 \
    -d imagenet