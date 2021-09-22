
srun -p train --cpus-per-task=96 -t 128:00:00 --gpus-per-node=8 \
  python parallel_train.py \
    --dataset ljspeech_nvidia \
    --epochs 8000 \
    --batch-size 128 \
    --workers 12 \
    --learning-rate 1e-4 \
    --n-freq 80 \
    --loss 'crossentropy' \
    --n-bits 8 \
    --checkpoint ./parallel_wavernn_nvidia_ckpt.pt