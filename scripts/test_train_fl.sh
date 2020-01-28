CUDA_VISIBLE_DEVICES=0 python train_fl.py models/mobilenet/feminst_large_test ALL data/ \
    -ge 100 -le 2 -lr 0.001 \
    -d feminst