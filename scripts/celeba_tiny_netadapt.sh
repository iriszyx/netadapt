CUDA_VISIBLE_DEVICES=0,1,2,3 python master.py models/celebanet/celeba_tiny_netadapt 3 128 128 \
    -im models/celebanet/celeba_model_tiny.pth.tar -gp 0 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.05 -rd 0.99 \
    -lr 0.001 -st 1  \
    -dp data/ --arch mobilenetfed -dn 1 -d celeba_tiny -gn 1