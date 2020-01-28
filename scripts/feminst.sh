CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python master.py models/mobilenet/feminst_mac_g14_d3500 3 224 224 \
    -im models/mobilenet/feminst_model.pth.tar -gp 0 1 2 3 4 5 6 7\
    -mi 15 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.98 \
    -lr 0.001 -st 1  \
    -dp data/ --arch mobilenetfed -dn 3500 -d feminst -gn 14
