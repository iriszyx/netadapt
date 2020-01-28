CUDA_VISIBLE_DEVICES=0,1,2,3 python master.py models/mobilenet/celeba_netadapt 3 224 224 \
    -im models/mobilenet/celeba_model.pth.tar -gp 0 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.05 -rd 0.98 \
    -lr 0.001 -st 1  \
    -dp data/ --arch mobilenetfed -dn 1 -d celeba -gn 1