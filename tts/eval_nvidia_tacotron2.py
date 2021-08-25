"""
Text-to-speech pipeline using Tacotron2.
"""

from functools import partial
import random

import torch
import torchaudio
from torch.utils.data import DataLoader
import numpy as np
import librosa
from tqdm import tqdm

from datasets import (
    text_mel_collate_fn,
    split_process_dataset,
    SpectralNormalization,
    InverseSpectralNormalization,
)
from text.text_preprocessing import (
    available_symbol_set,
    available_phonemizers,
    get_symbol_list,
    text_to_sequence,
)


def get_datasets():
    utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
    text_preprocessor = lambda x: utils.prepare_input_sequence([x], cpu_run=True)[0][0]

    transforms = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            win_length=1024,
            hop_length=256,
            f_min=0.0,
            f_max=8000.0,
            n_mels=80,
            mel_scale='slaney',
            normalized=False,
            power=1,
            norm='slaney',
        ),
        SpectralNormalization()
    )
    trainset, valset = split_process_dataset(
        'ljspeech_nvidia', "./", 0.1, transforms, text_preprocessor)
    return trainset, valset


def to_gpu(x):
    x = x.contiguous()
    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return x


def batch_to_gpu(batch):
    text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths, gate_padded = batch
    text_padded = to_gpu(text_padded).long()
    text_lengths = to_gpu(text_lengths).long()
    mel_specgram_padded = to_gpu(mel_specgram_padded).float()
    gate_padded = to_gpu(gate_padded).float()
    mel_specgram_lengths = to_gpu(mel_specgram_lengths).long()
    x = (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths)
    y = (mel_specgram_padded, gate_padded)
    return x, y



def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
    tacotron2 = tacotron2.to('cuda')
    tacotron2.eval()

    _, valset = get_datasets()
    loader_params = {
        "batch_size": 32,
        "num_workers": 8,
        "prefetch_factor": 1024,
        'persistent_workers': True,
        "shuffle": False,
        "pin_memory": True,
        "drop_last": False,
        "collate_fn": partial(text_mel_collate_fn, n_frames_per_step=1),
    }
    val_loader = DataLoader(valset, **loader_params)

    # reference from https://github.com/SamuelBroughton/Mel-Cepstral-Distortion/blob/master/mel-cepstral-distortion.ipynb
    def log_spec_dB_dist(x, y):
        log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
        diff = x - y
        
        return log_spec_dB_const * np.sqrt(np.inner(diff, diff))
    
    inv_norm = InverseSpectralNormalization()

    costs = []
    n_frames = []
    for batch in tqdm(val_loader):
        (text_padded, text_lengths, mel_specgram_padded, mel_specgram_lengths), y = batch_to_gpu(batch)
        with torch.no_grad():
            pred_mel_specgram, pred_mel_lengths, _ = tacotron2.infer(text_padded, text_lengths)

        pred_mel_specgram = inv_norm(pred_mel_specgram).detach().cpu().numpy()
        pred_mel_lengths = pred_mel_lengths.detach().cpu().numpy()
        mel_specgram_padded = inv_norm(mel_specgram_padded).detach().cpu().numpy()
        mel_specgram_lengths = mel_specgram_lengths.detach().cpu().numpy()
        for i in range(len(mel_specgram_padded)):
            min_cost, _ = librosa.sequence.dtw(
                mel_specgram_padded[i, :, :mel_specgram_lengths[i]],
                pred_mel_specgram[i, :, :pred_mel_lengths[i]],
                metric=log_spec_dB_dist)
            costs.append(np.mean(min_cost))
            n_frames.append(mel_specgram_lengths[i])
    
    print(np.sum(costs) / np.sum(n_frames))



if __name__ == "__main__":
    main()