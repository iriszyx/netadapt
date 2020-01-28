CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python master.py models/mobilenet/prune-by-latency_fl_mac 3 224 224 \
    -im models/mobilenet/model.pth.tar -gp 0 1 \
    -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 -lr 0.001 \
    -mi 30 -st 5 \
    -dp data/ --arch mobilenetfed -dn 140 -d cifar10 -gn 14


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python master.py models/mobilenet/prune-by-latency_fl 3 224 224 \
#     -im models/mobilenet/model.pth.tar -gp 0 1 \
#     -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
#     -lr 0.001 -lt latency_lut/lut_mobilenet.pkl \
#     -mi 35 -st 10 \
#     -dp data/ --arch mobilenetfed -dn 140 -d cifar10 -gn 14 -re


# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python master.py models/mobilenet/prune-by-latency_fl 3 224 224 \
#     -im models/mobilenet/model.pth.tar -gp 0 1 \
#     -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
#     -lr 0.001 -lt latency_lut/lut_mobilenet.pkl \
#     -mi 5 -st 5 \
#     -dp data/ --arch mobilenetfed -dn 2 -d cifar10 -gn 2
