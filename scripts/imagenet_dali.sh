CUDA_VISIBLE_DEVICES=3 python master_fl.py models/mobilenet/fl_imagenet_mac_g14_d1400_dali 3 224 224 \
    -im models/mobilenet/imagenet_model.pth.tar -gp 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.05 -rd 0.96 \
    -lr 0.0045 -st 3 \
    -dp /data1/dataset --arch mobilenet_imagenet -dn 1400 -d imagenet_dali -gn 14