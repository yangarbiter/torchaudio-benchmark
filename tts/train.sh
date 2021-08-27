
#srun -p train --cpus-per-task=96 -t 48:00:00 --gpus-per-node=8 \
#  python train.py \
#    --dataset ljspeech_nvidia \
#    --learning-rate 1e-3 \
#    --epochs 1501 --anneal-steps 500 1000 1500 --anneal-factor 0.1 \
#    --batch-size 96 --weight-decay 1e-6 --grad-clip 1.0 --text-preprocessor english_characters \
#    --logging-dir ./logs --checkpoint-path ./ckpt.pth --dataset-path ./

srun -p dev --cpus-per-task=96 -t 48:00:00 --gpus-per-node=8 \
  python train.py \
    --dataset ljspeech_nvidia \
    --workers 12 \
    --learning-rate 1e-3 \
    --epochs 1501 \
    --anneal-steps 500 1000 1500 \
    --anneal-factor 0.1 \
    --batch-size 96 \
    --weight-decay 1e-6 \
    --grad-clip 1.0 \
    --text-preprocessor english_phonemes \
    --phonemizer DeepPhonemizer \
    --phonemizer-checkpoint ./en_us_cmudict_forward.pt \
    --cmudict-root ./ \
    --logging-dir ./english_phonemes_logs \
    --checkpoint-path ./english_phonemes_ckpt.pth \
    --dataset-path ./
