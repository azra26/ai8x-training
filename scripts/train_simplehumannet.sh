#!/bin/sh 
#2022.05.25-142029 no augment

#python train.py --epochs 30 --optimizer Adam --lr 0.0001 --weight-decay 0.05 --compress policies/schedule-simplehumannet.yaml --enable-tensorboard --pr-curves --model ai85simplehumannet --dataset humans_vs_robots_simplenet --device MAX78000 --batch-size 128 --print-freq 100 --confusion --qat-policy policies/qat_policy_simplehumannet.yaml --param-hist --use-bias "$@"

python train.py --epochs 30 --optimizer Adam --lr 0.0001 --weight-decay 0.05 --compress policies/schedule-simplehumannet.yaml --enable-tensorboard --pr-curves --model ai85simplehumannetv2 --dataset humans_vs_robotsv2 --device MAX78000 --batch-size 128 --print-freq 100 --confusion --qat-policy policies/qat_policy_simplehumannetv2.yaml --param-hist --use-bias "$@"

