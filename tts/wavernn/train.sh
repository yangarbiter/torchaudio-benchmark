#!/usr/bin/bash

#srun -p dev --cpus-per-task=96 -t 48:00:00 --gpus-per-node=4 \
#  python main.py \
#    --batch-size 256 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 10

#srun -p dev --cpus-per-task=96 -t 48:00:00 --gpus-per-node=8 \
#  python main.py \
#    --batch-size 1024 \
#    --workers 48 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 10 \
#    --checkpoint ./wavernn_ckpt_v2.pt

#srun -p train --cpus-per-task=96 -t 48:00:00 --gpus-per-node=8 \
#  python main.py \
#    --dataset ljspeech_fatchord \
#    --batch-size 1024 \
#    --workers 48 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 9 \
#    --checkpoint ./wavernn_fatchord_ckpt.pt

srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
  python main.py \
    --dataset ljspeech_fatchord \
    --batch-size 2048 \
    --workers 96 \
    --learning-rate 1e-4 \
    --n-freq 80 \
    --loss 'crossentropy' \
    --n-bits 9 \
    --checkpoint ./wavernn_fatchord_ckpt_v2.pt