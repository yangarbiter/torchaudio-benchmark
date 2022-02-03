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

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python main.py \
#    --dataset ljspeech_fatchord \
#    --batch-size 1024 \
#    --workers 96 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./wavernn_fatchord_ckpt_v2.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python main.py \
#    --dataset ljspeech_fatchord \
#    --batch-size 256 \
#    --workers 96 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./wavernn_fatchord_ckpt_v3.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python main.py \
#    --dataset ljspeech_fatchord \
#    --epochs 2500 \
#    --batch-size 256 \
#    --workers 96 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'mol' \
#    --n-bits 8 \
#    --checkpoint ./wavernn_fatchord_ckpt_mol.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 8000 \
#    --batch-size 256 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 8000 \
#    --batch-size 32 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v2.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 10000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt.pt


# Above without norm="slaney"

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 8000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v3.pt


#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 10000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 9 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v4.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 10000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v5.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 10000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 10 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v6.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 6000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 9 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v7.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 6000 \
#    --batch-size 64 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 9 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v8.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 8000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss mol \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_mol.pt

#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_fatchord \
#    --epochs 8000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_fatchord_ckpt_v3.pt
#
#srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
#  python parallel_train.py \
#    --dataset ljspeech_nvidia \
#    --epochs 8000 \
#    --batch-size 128 \
#    --workers 12 \
#    --learning-rate 1e-4 \
#    --n-freq 80 \
#    --loss 'crossentropy' \
#    --n-bits 8 \
#    --checkpoint ./parallel_wavernn_nvidia_ckpt_v3.pt

python parallel_train.py \
  --dataset ljspeech_fatchord \
  --epochs 8000 \
  --batch-size 1 \
  --workers 1 \
  --learning-rate 1e-4 \
  --n-freq 80 \
  --loss 'crossentropy' \
  --n-bits 8 \
  --checkpoint ./parallel_wavernn_fatchord_ckpt_v3.pt
