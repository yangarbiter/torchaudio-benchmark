import random

import torch
from torchaudio.transforms import GriffinLim, MelSpectrogram, Resample, InverseMelScale
from tqdm import tqdm
import numpy as np
import joblib

from pesq import pesq
from pystoi import stoi

from eval_utils import eval_results, get_dataset


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sample_rate = 22050

    mel_kwargs = {
        'sample_rate': sample_rate,
        'n_fft': 2048,
        'f_min': 40.,
        'n_mels': 80,
        'win_length': 1100,
        'hop_length': 275,
        'mel_scale': 'slaney',
        'norm': 'slaney',
        'power': 1,
    }
    spec_transform = MelSpectrogram(**mel_kwargs)

    vocoder = torch.nn.Sequential(
        InverseMelScale(
            n_stft=(2048 // 2 + 1),
            n_mels=80,
            sample_rate=sample_rate,
            f_min=40.,
            mel_scale="slaney",
            norm='slaney',
        ),
        GriffinLim(
            n_fft=2048,
            power=1,
            hop_length=275,
            win_length=1100,
            n_iter=32,
        )
    ).to(device)

    dset, _ = get_dataset()

    preds = []
    for i in tqdm(range(50)):
        specgram = spec_transform(dset[i][0])
        preds.append(vocoder(specgram.to(device)).cpu())

    eval_results(preds, dset, sample_rate)


if __name__ == "__main__":
    main()
