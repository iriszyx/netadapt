CUDA_VISIBLE_DEVICES=1,2,3 python master.py models/mobilenet/celeba 3 224 224 \
    -im feminst_model.pth.tar -gp 1 2 3 \
    -mi 10 -bur 0.25 -rt FLOPS  -irr 0.025 -rd 0.97 \
    -lr 0.001 -st 1  \
    -dp data/ --arch mobilenetfed -dn 28 -d celeba -gn 14
