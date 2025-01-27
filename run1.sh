#!/bin/bash

T=`date +%m%d%H%M%S`

mkdir exp
mkdir exp/$T
#mkdir exp/$T/code
#cp -r datasets exp/$T/code/datasets
#cp -r models exp/$T/code/models
#cp ./*.py exp/$T/code/
#cp run.sh exp/$T/code

mkdir exp/$T/train.log

datapath=D:/pycharmprojects/few shot learning/counting/SPDCN-CAC-main/data_list

#python main.py --data-path $datapath --batch-size 4 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
python main.py --data-path D:/pycharmprojects/few-shot-learning/counting/SPDCN-CAC-main/data_list --batch-size 1 --eval --resume D:/pycharmprojects/few-shot-learning/counting/FFMC1/exp/0305103929/output/ckpt_epoch_best.pth --tag $T 2>&1 | tee exp/$T/train.log/running.log
