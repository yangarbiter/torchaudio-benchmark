
#gpui python eval_tacotron2.py --checkpoint-path=./ckpt.pth --dataset ljspeech_nvidia

#gpui python eval_nvidia_tacotron2.py

#gpui python eval_tacotron2.py \
#  --text-preprocessor english_phonemes \
#  --dataset ljspeech_nvidia \
#  --checkpoint-path ./english_phonemes_ckpt.pth

gpui python eval_tacotron2.py --checkpoint-path=./best_ckpt.pth --dataset ljspeech_nvidia

# 2.283412922933646
#gpui python eval_tacotron2.py \
#  --text-preprocessor english_phonemes \
#  --dataset ljspeech_nvidia \
#  --checkpoint-path ./best_english_phonemes_ckpt.pth


#gpui python eval_tacotron2.py \
#  --dataset ljspeech \
#  --checkpoint-name=tacotron2_english_characters_1500_epochs_ljspeech
