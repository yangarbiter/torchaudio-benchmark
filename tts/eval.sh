
#gpui python eval_tacotron2.py --checkpoint-path=./ckpt.pth --dataset ljspeech_nvidia

#gpui python eval_nvidia_tacotron2.py

gpui python eval_tacotron2.py \
  --text-preprocessor english_phonemes \
  --dataset ljspeech \
  --checkpoint-name=tacotron2_english_phonemes_1500_epochs_ljspeech


#gpui python eval_tacotron2.py \
#  --dataset ljspeech \
#  --checkpoint-name=tacotron2_english_characters_1500_epochs_ljspeech