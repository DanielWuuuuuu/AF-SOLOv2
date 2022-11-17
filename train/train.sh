#!/bin/bash


source /etc/profile

if [ -f "/home/yuanxue/anaconda3/etc/profile.d/conda.sh" ]; then
    . "/home/yuanxue/anaconda3/etc/profile.d/conda.sh"
 else
     export PATH="/home/yuanxue/anaconda3/bin:$PATH"
 fi

cd /media/yuanxue/mine/wy/SOLO-master/train && conda activate solov2 && python train.py
