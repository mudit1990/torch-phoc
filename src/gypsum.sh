#!/bin/sh

python train.py --dataset maps --batch_size 32 -lrs 6000:1e-3,10000:1e-4 --gpu_id 0
