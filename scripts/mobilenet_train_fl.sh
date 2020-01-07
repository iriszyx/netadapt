CUDA_VISIBLE_DEVICES=0 python train_fl.py models/mobilenet/prune-by-latency_fl iter_15_block_7_model.pth.tar data/ \
    -ge 100 -le 2 -lr 0.001 \
    -d cifar10