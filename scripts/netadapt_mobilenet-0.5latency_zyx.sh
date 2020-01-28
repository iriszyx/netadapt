CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 python master.py models/mobilenet/prune-by-latency 3 224 224 \
    -im models/mobilenet/model.pth.tar -gp 0 1 2 3 \
    -mi 20 -bur 0.25 -rt LATENCY  -irr 0.025 -rd 0.96 \
    -lr 0.001 -st 10 -lt latency_lut/lut_mobilenet.pkl \
    -dp data/ --arch mobilenetfed -dn 28 -d cifar10 -gn 14
