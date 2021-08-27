
srun -p train --cpus-per-task=48 -t 48:00:00 --gpus-per-node=8 \
  python train_wavernn.py
