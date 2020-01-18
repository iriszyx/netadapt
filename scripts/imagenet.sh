CUDA_VISIBLE_DEVICES=0,1,2 python master.py models/mobilenet/imagenet_mac_g14_d1400 3 224 224 \
    -im models/mobilenet/imagenet_model.pth.tar -gp 0 1 2 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.1 -rd 0.98 \
    -lr 0.1 -st 1  \
    -dp /data1/datas --arch mobilenetfed -dn 1400 -d imagenet -gn 14