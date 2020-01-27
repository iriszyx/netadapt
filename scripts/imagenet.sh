CUDA_VISIBLE_DEVICES=1,2,3 python master.py models/mobilenet/acc_imagenet_mac_g5_d700 3 224 224 \
    -im models/mobilenet/imagenet_model.pth.tar -gp 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.05 -rd 0.96 \
    -lr 0.0045 -st 1 \
    -dp /data1/dataset --arch mobilenet_imagenet -dn 700 -d imagenet -gn 5