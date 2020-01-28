CUDA_VISIBLE_DEVICES=0,1,2,3 python master.py models/mobilenet/imagenet_netadapt 3 224 224 \
    -im models/mobilenet/imagenet_model.pth.tar -gp 0 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.96 \
    -lr 0.1 -st 0.06 \
    -dp data/ --arch mobilenet -dn 1 -d imagnet -gn 1