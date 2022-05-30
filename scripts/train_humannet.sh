#!/bin/sh
python train.py --epochs 100 --optimizer Adam --lr 0.001 --wd 0 --deterministic --compress policies/schedule-humannet.yaml --model ai85humannet --dataset humans_vs_robots --batch-size 100 --device MAX78000 --enable-tensorboard --pr-curves --confusion --param-hist --embedding "$@"
