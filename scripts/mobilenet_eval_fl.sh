CUDA_VISIBLE_DEVICES=0 python eval_fl.py \
    models/mobilenet/prune-by-latency_fl \
    ALL \
    data/ \
    -d cifar10