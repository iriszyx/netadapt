CUDA_VISIBLE_DEVICES=1,2,3 python master.py models/mobilenet/celeba_mac_g14_d9100_st4 3 224 224 \
    -im models/mobilenet/celeba_model.pth.tar -gp 1 2 3 \
    -mi 20 -bur 0.25 -rt FLOPS  -irr 0.05 -rd 0.98 \
    -lr 0.001 -st 4  \
    -dp data/ --arch mobilenetfed -dn 9100 -d celeba -gn 14
