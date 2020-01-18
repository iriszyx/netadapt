CUDA_VISIBLE_DEVICES=0,1,2,3 python master.py models/mobilenet/imagenet_netadapt45 3 224 224 \
    -im models/mobilenet/imagenet_model.pth.tar -gp 0 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.1 -rd 0.96 \
    -lr 0.045 -st 1 \
    -dp /data1/dataset --arch mobilenetfed -dn 1 -d imagenet -gn 1