#!/bin/sh
python train.py --model ai85humannet --dataset humans_vs_robots --confusion --evaluate --exp-load-weights-from ../GTC_Hackaton_Models/faceid_net/model_052622/synthesizer_input/trained/ai85-humannet-qat_best.pth.tar -8 --save-sample 10 --device MAX78000 "$@"
