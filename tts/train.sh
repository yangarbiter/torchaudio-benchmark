
srun -p train --cpus-per-task=96 -t 48:00:00 --gpus-per-node=8 \
  python train.py --learning-rate 1e-3 --epochs 1501 --anneal-steps 500 1000 1500 --anneal-factor 0.1 \
  --batch-size 96 --weight-decay 1e-6 --grad-clip 1.0 --text-preprocessor english_characters \
  --logging-dir ./logs --checkpoint-path ./ckpt.pth --dataset-path ./